/**
 * Obedient Beast - WhatsApp Bridge
 * =================================
 * Connects to WhatsApp via Baileys and forwards messages to the Python server.
 * 
 * Usage:
 *   npm install
 *   node bridge.js
 * 
 * First run: Scan the QR code with WhatsApp (Settings > Linked Devices)
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
const AUTH_DIR = 'auth_info'
const BACKUP_DIR = path.join(os.homedir(), '.beast_whatsapp_backup')

// Store for pending responses
let sock = null

// ---------------------------------------------------------------------------
// Auto-Backup/Restore for WhatsApp credentials
// ---------------------------------------------------------------------------
// Automatically backs up auth_info after successful connection and
// restores from backup if auth_info is missing. No manual steps needed!
// ---------------------------------------------------------------------------

function backupAuth() {
    if (!fs.existsSync(AUTH_DIR)) return
    
    try {
        // Create backup directory if needed
        if (!fs.existsSync(BACKUP_DIR)) {
            fs.mkdirSync(BACKUP_DIR, { recursive: true })
        }
        
        // Copy all files from auth_info to backup
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
    // Only restore if auth_info is missing but backup exists
    if (fs.existsSync(AUTH_DIR) && fs.readdirSync(AUTH_DIR).length > 0) return false
    if (!fs.existsSync(BACKUP_DIR)) return false
    
    try {
        // Create auth_info if needed
        if (!fs.existsSync(AUTH_DIR)) {
            fs.mkdirSync(AUTH_DIR, { recursive: true })
        }
        
        // Copy all files from backup to auth_info
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
    
    // Load auth state (persisted across restarts)
    const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR)
    
    // Get latest Baileys version
    const { version } = await fetchLatestBaileysVersion()
    console.log(`Using Baileys v${version.join('.')}`)
    
    // Create socket connection
    sock = makeWASocket({
        version,
        auth: state,
        printQRInTerminal: false,  // We'll handle QR ourselves
    })
    
    // Handle connection updates
    sock.ev.on('connection.update', async (update) => {
        const { connection, lastDisconnect, qr } = update
        
        if (qr) {
            // Display QR code in terminal
            console.log('\n📱 Scan this QR code with WhatsApp:\n')
            qrcode.generate(qr, { small: true })
            console.log('\n(WhatsApp > Settings > Linked Devices > Link a Device)\n')
        }
        
        if (connection === 'close') {
            const reason = lastDisconnect?.error?.output?.statusCode
            console.log(`Connection closed. Reason: ${reason}`)
            
            // Reconnect unless logged out
            if (reason !== DisconnectReason.loggedOut) {
                console.log('Reconnecting...')
                setTimeout(connectToWhatsApp, 3000)
            } else {
                console.log('Logged out. Delete auth_info folder and restart to re-link.')
            }
        }
        
        if (connection === 'open') {
            console.log('✅ Connected to WhatsApp!')
            console.log(`🐺 Obedient Beast ready. Forwarding to ${BEAST_URL}`)
            
            // Auto-backup credentials after successful connection
            setTimeout(backupAuth, 2000)  // Wait for creds to be saved first
        }
    })
    
    // Save credentials when updated
    sock.ev.on('creds.update', saveCreds)
    
    // Handle incoming messages
    sock.ev.on('messages.upsert', async ({ messages, type }) => {
        // Only process new messages (not history sync)
        if (type !== 'notify') return
        
        // Get our own user ID for self-message detection
        const myId = sock.user?.id?.split(':')[0] || ''
        
        for (const msg of messages) {
            const chatId = msg.key.remoteJid
            const isGroup = chatId.endsWith('@g.us')
            
            // For groups, get the actual sender; for DMs, it's the chat ID
            let sender = chatId
            let isFromMe = msg.key.fromMe
            
            if (isGroup) {
                // In groups, participant is the actual sender
                const participant = msg.key.participant || ''
                sender = participant
                
                // Check if this is our own message in a group (via lid or phone)
                // This catches messages we send from our phone appearing on linked device
                if (participant.includes(myId) || participant.endsWith('@lid')) {
                    // This might be us - mark as potentially from owner
                    sender = 'OWNER'
                }
            }
            
            // Skip messages we sent (will show as fromMe on some devices)
            if (isFromMe && !isGroup) continue
            
            // Get message text
            const text = msg.message?.conversation || 
                         msg.message?.extendedTextMessage?.text ||
                         null
            
            if (!text) continue  // Skip non-text messages
            
            // Log with chat ID so user can identify which chat to allow
            console.log(`\n[${isGroup ? 'GROUP' : 'DM'}][${chatId}][${sender}] ${text.substring(0, 50)}...`)
            
            try {
                // Send to Python server (sender and chat_id used for auth check)
                const response = await fetch(`${BEAST_URL}/message`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, sender, chat_id: chatId })
                })
                
                if (!response.ok) {
                    // 403 = not authorized, silently ignore
                    if (response.status === 403) {
                        console.log(`[Blocked] chat=${chatId} sender=${sender}`)
                        continue
                    }
                    console.error(`Server error: ${response.status}`)
                    continue
                }
                
                const data = await response.json()
                const reply = data.response
                const imagePath = data.image
                
                if (reply) {
                    console.log(`[Reply] ${reply.substring(0, 100)}...`)
                    
                    // Send reply back to the chat (group or DM)
                    await sock.sendMessage(chatId, { text: reply })
                }
                
                // Send image if provided
                if (imagePath && fs.existsSync(imagePath)) {
                    console.log(`[Image] Sending ${imagePath}`)
                    await sock.sendMessage(chatId, { 
                        image: fs.readFileSync(imagePath),
                        caption: 'Screenshot'
                    })
                }
                
            } catch (error) {
                console.error('Error calling Beast server:', error.message)
                
                // Notify user of error in the chat
                await sock.sendMessage(chatId, { 
                    text: '⚠️ Beast server unavailable. Please try again later.' 
                })
            }
        }
    })
}

// Start
console.log('=' .repeat(60))
console.log('🐺 Obedient Beast - WhatsApp Bridge')
console.log('=' .repeat(60))
connectToWhatsApp()
