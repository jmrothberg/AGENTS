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

// Configuration
const BEAST_URL = process.env.BEAST_URL || 'http://localhost:5001'

// Store for pending responses
let sock = null

async function connectToWhatsApp() {
    // Load auth state (persisted across restarts)
    const { state, saveCreds } = await useMultiFileAuthState('auth_info')
    
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
            
            console.log(`\n[${isGroup ? 'GROUP' : 'DM'}][${sender}] ${text}`)
            
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
                        console.log(`[Blocked] ${sender} not authorized`)
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
