/**
 * Obedient Beast - WhatsApp Bridge
 * =================================
 * Connects to WhatsApp via Baileys and forwards messages to the Python server.
 *
 * Architecture:
 *   WhatsApp Cloud → Baileys (this script) → HTTP POST → server.py → beast.run()
 *   Response flows back: beast.run() → server.py → this script → WhatsApp
 *
 * Baileys library:
 *   @whiskeysockets/baileys is a JavaScript library that implements the
 *   WhatsApp Web protocol. It connects to WhatsApp's servers the same way
 *   the WhatsApp Web browser client does. This means Beast uses YOUR
 *   existing WhatsApp account (not a separate bot account).
 *
 * Auth flow:
 *   1. First run: Baileys generates a QR code displayed in the terminal
 *   2. User scans QR with their phone (WhatsApp > Settings > Linked Devices)
 *   3. Baileys saves auth credentials to auth_info/ directory
 *   4. Subsequent runs: auto-connects using saved credentials (no QR needed)
 *   5. Credentials are auto-backed up to ~/.beast_whatsapp_backup/ for safety
 *
 * OWNER detection for group messages:
 *   In group chats, messages from the phone running WhatsApp appear with
 *   a special participant ID (often ending in @lid or matching sock.user.id).
 *   We detect these and tag them as "OWNER" so server.py always allows them.
 *
 * Usage:
 *   npm install
 *   node bridge.js
 *
 *   First run: Scan the QR code with WhatsApp (Settings > Linked Devices)
 */

import makeWASocket, {
    useMultiFileAuthState,
    DisconnectReason,
    fetchLatestBaileysVersion
} from '@whiskeysockets/baileys'
import qrcode from 'qrcode-terminal'
import fs from 'fs'
import path from 'path'
import os from 'os'

// Configuration
const BEAST_URL = process.env.BEAST_URL || 'http://localhost:5001'
const AUTH_DIR = 'auth_info'                                     // Active credentials (gitignored)
const BACKUP_DIR = path.join(os.homedir(), '.beast_whatsapp_backup')  // Auto-backup location

// Store for the WhatsApp socket connection
let sock = null

// ---------------------------------------------------------------------------
// Auto-Backup/Restore for WhatsApp credentials
// ---------------------------------------------------------------------------
// Automatically backs up auth_info after successful connection and
// restores from backup if auth_info is missing. This means you can
// delete auth_info/ and it auto-restores, or move to a new machine
// by copying ~/.beast_whatsapp_backup/ and it just works.
// ---------------------------------------------------------------------------

function backupAuth() {
    // Only backup if auth_info directory exists and has files
    if (!fs.existsSync(AUTH_DIR)) return

    try {
        // Create backup directory if needed
        if (!fs.existsSync(BACKUP_DIR)) {
            fs.mkdirSync(BACKUP_DIR, { recursive: true })
        }

        // Copy all credential files from auth_info to backup
        const files = fs.readdirSync(AUTH_DIR)
        for (const file of files) {
            fs.copyFileSync(
                path.join(AUTH_DIR, file),
                path.join(BACKUP_DIR, file)
            )
        }
        console.log(`📦 Auth backed up to ${BACKUP_DIR}`)
    } catch (err) {
        console.error('Backup failed:', err.message)
    }
}

function restoreAuth() {
    // Only restore if auth_info is missing/empty but backup exists
    if (fs.existsSync(AUTH_DIR) && fs.readdirSync(AUTH_DIR).length > 0) return false
    if (!fs.existsSync(BACKUP_DIR)) return false

    try {
        // Create auth_info if needed
        if (!fs.existsSync(AUTH_DIR)) {
            fs.mkdirSync(AUTH_DIR, { recursive: true })
        }

        // Copy all credential files from backup to auth_info
        const files = fs.readdirSync(BACKUP_DIR)
        for (const file of files) {
            fs.copyFileSync(
                path.join(BACKUP_DIR, file),
                path.join(AUTH_DIR, file)
            )
        }
        console.log(`✅ Auth restored from backup!`)
        return true
    } catch (err) {
        console.error('Restore failed:', err.message)
        return false
    }
}

async function connectToWhatsApp() {
    // Try to restore from backup if auth_info is missing
    restoreAuth()

    // Load auth state (persisted across restarts via auth_info/ directory)
    const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR)

    // Get latest Baileys version (protocol compatibility)
    const { version } = await fetchLatestBaileysVersion()
    console.log(`Using Baileys v${version.join('.')}`)

    // Create WhatsApp Web socket connection
    sock = makeWASocket({
        version,
        auth: state,
        printQRInTerminal: false,  // We'll handle QR ourselves for better formatting
    })

    // Handle connection lifecycle events
    sock.ev.on('connection.update', async (update) => {
        const { connection, lastDisconnect, qr } = update

        // Display QR code when WhatsApp needs device linking
        if (qr) {
            console.log('\n📱 Scan this QR code with WhatsApp:\n')
            qrcode.generate(qr, { small: true })
            console.log('\n(WhatsApp > Settings > Linked Devices > Link a Device)\n')
        }

        if (connection === 'close') {
            const reason = lastDisconnect?.error?.output?.statusCode
            console.log(`Connection closed. Reason: ${reason}`)

            // Reconnect unless explicitly logged out
            // DisconnectReason.loggedOut means the user unlinked the device
            if (reason !== DisconnectReason.loggedOut) {
                console.log('Reconnecting...')
                setTimeout(connectToWhatsApp, 3000)  // Wait 3s before retry
            } else {
                console.log('Logged out. Delete auth_info folder and restart to re-link.')
            }
        }

        if (connection === 'open') {
            console.log('✅ Connected to WhatsApp!')
            console.log(`🐺 Obedient Beast ready. Forwarding to ${BEAST_URL}`)

            // Auto-backup credentials after successful connection.
            // Wait 2s for Baileys to finish saving credential files first.
            setTimeout(backupAuth, 2000)
        }
    })

    // Save credentials when updated (Baileys rotates keys periodically)
    sock.ev.on('creds.update', saveCreds)

    // Handle incoming messages
    sock.ev.on('messages.upsert', async ({ messages, type }) => {
        // Only process new messages (type='notify'), not history sync
        if (type !== 'notify') return

        // Get our own user ID for self-message detection
        // sock.user.id format: "1234567890:12@s.whatsapp.net" — we extract the number part
        const myId = sock.user?.id?.split(':')[0] || ''

        for (const msg of messages) {
            const chatId = msg.key.remoteJid        // Chat ID (phone@s.whatsapp.net or group@g.us)
            const isGroup = chatId.endsWith('@g.us') // Group chat detection

            // Determine the actual sender
            let sender = chatId
            let isFromMe = msg.key.fromMe

            if (isGroup) {
                // In groups, msg.key.participant is the actual sender's ID
                const participant = msg.key.participant || ''
                sender = participant

                // OWNER detection in groups:
                // Messages sent from our phone appear with participant containing
                // our phone number or ending with @lid (linked device ID).
                // We tag these as "OWNER" so server.py always authorizes them.
                if (participant.includes(myId) || participant.endsWith('@lid')) {
                    sender = 'OWNER'
                }
            }

            // Skip our own messages in DMs (they show as fromMe=true).
            // In groups, we let OWNER messages through for Beast processing.
            if (isFromMe && !isGroup) continue

            // Get message text (WhatsApp has two text formats)
            const text = msg.message?.conversation ||            // Simple text message
                         msg.message?.extendedTextMessage?.text || // Reply/forwarded text
                         null

            if (!text) continue  // Skip non-text messages (images, stickers, etc.)

            // Log with chat ID so user can identify which chat to add to ALLOWED_GROUPS
            console.log(`\n[${isGroup ? 'GROUP' : 'DM'}][${chatId}][${sender}] ${text.substring(0, 50)}...`)

            try {
                // Forward message to Python server (server.py)
                // sender and chat_id are used for authorization checks
                const response = await fetch(`${BEAST_URL}/message`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, sender, chat_id: chatId })
                })

                if (!response.ok) {
                    // 403 = not authorized — silently ignore (expected for non-allowed senders)
                    if (response.status === 403) {
                        console.log(`[Blocked] chat=${chatId} sender=${sender}`)
                        continue
                    }
                    console.error(`Server error: ${response.status}`)
                    continue
                }

                const data = await response.json()
                const reply = data.response
                const imagePath = data.image  // Optional: screenshot path from Beast

                // Send text reply back to the WhatsApp chat
                if (reply) {
                    console.log(`[Reply] ${reply.substring(0, 100)}...`)
                    await sock.sendMessage(chatId, { text: reply })
                }

                // Send image if Beast included one (e.g., from screenshot tool)
                if (imagePath && fs.existsSync(imagePath)) {
                    console.log(`[Image] Sending ${imagePath}`)
                    await sock.sendMessage(chatId, {
                        image: fs.readFileSync(imagePath),
                        caption: 'Screenshot'
                    })
                }

            } catch (error) {
                console.error('Error calling Beast server:', error.message)

                // Notify user of error in the WhatsApp chat
                await sock.sendMessage(chatId, {
                    text: '⚠️ Beast server unavailable. Please try again later.'
                })
            }
        }
    })
}

// Start the bridge
console.log('=' .repeat(60))
console.log('🐺 Obedient Beast - WhatsApp Bridge')
console.log('=' .repeat(60))
connectToWhatsApp()
