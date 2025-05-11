#include <WiFi.h>
#include <WebServer.h>
#include <Servo.h>
#include <EEPROM.h>
#include <LEAmDNS.h>

const char* ssid = "SindiWIFI";
const char* password = "SamsunG2016";

// Pins
const int relayPinLow = 17;    // Low speed relay
const int relayPinMedium = 19; // Medium speed relay
const int servoPin = 0;

// Create objects
WebServer server(80);
Servo servo;

// Structure for storing settings
struct Settings {
    int angleOpen;      // Open Angle (default 180)
    int angleClose;     // Close Angle (default 0)
    int anglePark;      // Park Angle (default 32)
} settings;

// State variables
bool relayLowState = false;
bool relayMediumState = false;
int currentAngle = 0;
bool isVentActive = false;
String ventSpeed = "off"; // "off", "low", "medium", "max"

// HTML page in English – updated for Multi-Speed Control
const char webPage[] PROGMEM = R"=====(
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta charset="UTF-8">
    <title>Ventilation Control</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .control-block {
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        h2 {
            color: #666;
            margin-bottom: 15px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
        }
        button:hover {
            background-color: #45a049;
        }
        button.off {
            background-color: #f44336;
        }
        button.off:hover {
            background-color: #da190b;
        }
        button.low {
            background-color: #2196F3;
        }
        button.low:hover {
            background-color: #0b7dda;
        }
        button.medium {
            background-color: #ff9800;
        }
        button.medium:hover {
            background-color: #e68a00;
        }
        button.max {
            background-color: #9c27b0;
        }
        button.max:hover {
            background-color: #7b1fa2;
        }
        button.selected {
            border: 3px solid #333;
            font-weight: bold;
        }
        .settings-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }
        .setting-item {
            text-align: center;
        }
        input[type="number"] {
            width: 80px;
            padding: 5px;
            margin: 5px 0;
        }
        .status {
            text-align: center;
            font-size: 18px;
            margin: 20px 0;
            padding: 10px;
            background-color: #e8f5e9;
            border-radius: 5px;
        }
        .speed-controls {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Ventilation Control</h1>
        
        <div class="status" id="statusDisplay">
            Status: <span id="ventStatus">Off</span> | 
            Speed: <span id="ventSpeed">Off</span>
        </div>

        <div class="control-block">
            <h2>Ventilation Controls</h2>
            <div class="speed-controls">
                <button onclick="setVentSpeed('off')" id="offButton" class="off selected">Off</button>
                <button onclick="setVentSpeed('low')" id="lowButton" class="low">Low</button>
                <button onclick="setVentSpeed('medium')" id="mediumButton" class="medium">Medium</button>
                <button onclick="setVentSpeed('max')" id="maxButton" class="max">Max</button>
            </div>
        </div>
        
        <div class="control-block">
            <h2>Angle Settings</h2>
            <div class="settings-grid">
                <div class="setting-item">
                    <label>Open Angle:</label>
                    <input type="number" id="angleOpen" min="0" max="180">
                </div>
                <div class="setting-item">
                    <label>Close Angle:</label>
                    <input type="number" id="angleClose" min="0" max="180">
                </div>
                <div class="setting-item">
                    <label>Park Angle:</label>
                    <input type="number" id="anglePark" min="0" max="180">
                </div>
            </div>
            <button onclick="saveSettings()">Save Settings</button>
        </div>
    </div>

    <script>
        function updateStatus() {
            fetch('/status')
            .then(response => response.json())
            .then(status => {
                document.getElementById('ventStatus').textContent = status.ventActive ? 'On' : 'Off';
                document.getElementById('ventSpeed').textContent = capitalizeFirstLetter(status.ventSpeed);
                
                // Update speed button selection
                document.getElementById('offButton').classList.remove('selected');
                document.getElementById('lowButton').classList.remove('selected');
                document.getElementById('mediumButton').classList.remove('selected');
                document.getElementById('maxButton').classList.remove('selected');
                
                document.getElementById(status.ventSpeed + 'Button').classList.add('selected');
                
                // Update angle settings fields
                document.getElementById('angleOpen').value = status.settings.angleOpen;
                document.getElementById('angleClose').value = status.settings.angleClose;
                document.getElementById('anglePark').value = status.settings.anglePark;
            });
        }

        function capitalizeFirstLetter(string) {
            return string.charAt(0).toUpperCase() + string.slice(1);
        }

        function setVentSpeed(speed) {
            fetch('/vent/' + speed)
            .then(response => response.text())
            .then(data => {
                updateStatus();
            });
        }

        function saveSettings() {
            const settings = {
                angleOpen: document.getElementById('angleOpen').value,
                angleClose: document.getElementById('angleClose').value,
                anglePark: document.getElementById('anglePark').value
            };
            fetch('/settings/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(settings)
            })
            .then(response => response.text())
            .then(data => alert('Settings saved'));
        }

        // Update status every 2 seconds
        setInterval(updateStatus, 2000);
        updateStatus();
    </script>
</body>
</html>
)=====";

void saveSettingsToEEPROM() {
    EEPROM.put(0, settings);
    EEPROM.commit();
}

void loadSettingsFromEEPROM() {
    EEPROM.get(0, settings);
    
    // Validate data and set default values if needed
    if (settings.angleOpen < 0 || settings.angleOpen > 180) settings.angleOpen = 180;
    if (settings.angleClose < 0 || settings.angleClose > 180) settings.angleClose = 0;
    if (settings.anglePark < 0 || settings.anglePark > 180) settings.anglePark = 32;
}

void startupSequence() {
    // Initial movement sequence
    servo.write(settings.angleOpen);
    delay(1000);
    servo.write(settings.angleClose);
    delay(2000);
    servo.write(settings.anglePark);
    currentAngle = settings.anglePark;
}

void setup() {
    Serial.begin(115200);
    
    // Initialize EEPROM
    EEPROM.begin(512);
    loadSettingsFromEEPROM();
    
    // Setup pins
    pinMode(relayPinLow, OUTPUT);
    pinMode(relayPinMedium, OUTPUT);
    digitalWrite(relayPinLow, HIGH);   // Initial state – off (relay modules are typically active LOW)
    digitalWrite(relayPinMedium, HIGH); // Initial state – off
    servo.attach(servoPin);
    
    // Execute startup sequence
    startupSequence();
    
    // Connect to WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.println("WiFi connected");
    Serial.println("IP address: ");
    Serial.println(WiFi.localIP());
    
    // Setup mDNS for service publishing, so Pi5 can find it
    if (!MDNS.begin("pico")) {
        Serial.println("Error setting up MDNS responder!");
    } else {
        Serial.println("mDNS responder started");
        MDNS.addService("_pico-vent", "tcp", 80);
    }
    
    // Setup web server routes
    server.on("/", HTTP_GET, handleRoot);
    server.on("/relay/toggle", HTTP_GET, handleRelayToggle);
    server.on("/servo/set", HTTP_GET, handleServoSet);
    server.on("/vent/low", HTTP_GET, handleVentLow);
    server.on("/vent/medium", HTTP_GET, handleVentMedium);
    server.on("/vent/max", HTTP_GET, handleVentMax);
    server.on("/vent/off", HTTP_GET, handleVentOff);
    server.on("/status", HTTP_GET, handleStatus);
    server.on("/settings/save", HTTP_POST, handleSettingsSave);
    
    server.begin();
    Serial.println("HTTP server started");
}

void loop() {
    server.handleClient();
    MDNS.update();
}

void handleRoot() {
    server.send(200, "text/html", webPage);
}

void handleRelayToggle() {
    // Toggle both relays simultaneously (for backward compatibility)
    if (relayLowState || relayMediumState) {
        // Turn off both
        digitalWrite(relayPinLow, HIGH);
        digitalWrite(relayPinMedium, HIGH);
        relayLowState = false;
        relayMediumState = false;
        ventSpeed = "off";
        isVentActive = false;
    } else {
        // Turn on both (max speed)
        digitalWrite(relayPinLow, LOW);
        digitalWrite(relayPinMedium, LOW);
        relayLowState = true;
        relayMediumState = true;
        ventSpeed = "max";
        isVentActive = true;
    }
    server.send(200, "text/plain", (relayLowState || relayMediumState) ? "1" : "0");
}

void handleServoSet() {
    if(server.hasArg("angle")) {
        int angle = server.arg("angle").toInt();
        if(angle >= 0 && angle <= 180) {
            servo.write(angle);
            currentAngle = angle;
            server.send(200, "text/plain", "OK");
        } else {
            server.send(400, "text/plain", "Invalid angle");
        }
    } else {
        server.send(400, "text/plain", "Missing angle parameter");
    }
}

void handleVentLow() {
    ventSpeed = "low";
    isVentActive = true;
    servo.write(settings.angleOpen);
    currentAngle = settings.angleOpen;
    
    // Set relays for low speed
    digitalWrite(relayPinLow, LOW);    // Turn on Low relay
    digitalWrite(relayPinMedium, HIGH); // Turn off Medium relay
    relayLowState = true;
    relayMediumState = false;
    
    server.send(200, "text/plain", "OK");
}

void handleVentMedium() {
    ventSpeed = "medium";
    isVentActive = true;
    servo.write(settings.angleOpen);
    currentAngle = settings.angleOpen;
    
    // Set relays for medium speed
    digitalWrite(relayPinLow, HIGH);   // Turn off Low relay
    digitalWrite(relayPinMedium, LOW); // Turn on Medium relay
    relayLowState = false;
    relayMediumState = true;
    
    server.send(200, "text/plain", "OK");
}

void handleVentMax() {
    ventSpeed = "max";
    isVentActive = true;
    servo.write(settings.angleOpen);
    currentAngle = settings.angleOpen;
    
    // Set relays for max speed (both on)
    digitalWrite(relayPinLow, LOW);    // Turn on Low relay
    digitalWrite(relayPinMedium, LOW); // Turn on Medium relay
    relayLowState = true;
    relayMediumState = true;
    
    server.send(200, "text/plain", "OK");
}

void handleVentOff() {
    isVentActive = false;
    ventSpeed = "off";
    
    // Turn off both relays
    digitalWrite(relayPinLow, HIGH);
    digitalWrite(relayPinMedium, HIGH);
    relayLowState = false;
    relayMediumState = false;
    
    servo.write(settings.angleClose);
    currentAngle = settings.angleClose;
    delay(2000);
    servo.write(settings.anglePark);
    currentAngle = settings.anglePark;
    
    server.send(200, "text/plain", "OK");
}

void handleStatus() {
    String json = "{\"ventActive\":" + String(isVentActive ? "true" : "false") + 
                 ",\"ventSpeed\":\"" + ventSpeed + "\"" +
                 ",\"relayLow\":" + String(relayLowState ? "true" : "false") + 
                 ",\"relayMedium\":" + String(relayMediumState ? "true" : "false") + 
                 ",\"currentAngle\":" + String(currentAngle) + 
                 ",\"settings\":{" +
                 "\"angleOpen\":" + String(settings.angleOpen) + 
                 ",\"angleClose\":" + String(settings.angleClose) + 
                 ",\"anglePark\":" + String(settings.anglePark) + 
                 "}}";
    server.send(200, "application/json", json);
}

void handleSettingsSave() {
    if (server.hasArg("plain")) {
        String json = server.arg("plain");
        
        // Create temporary variables for new values
        int newAngleOpen = settings.angleOpen;
        int newAngleClose = settings.angleClose;
        int newAnglePark = settings.anglePark;
        
        // Find values in JSON more reliably
        int startPos, endPos;
        
        // Parse angleOpen
        startPos = json.indexOf("\"angleOpen\":");
        if (startPos != -1) {
            startPos += 11; // length of "angleOpen":
            endPos = json.indexOf(",", startPos);
            if (endPos != -1) {
                String valueStr = json.substring(startPos, endPos);
                valueStr.trim();
                int value = valueStr.toInt();
                if (value >= 0 && value <= 180) {
                    newAngleOpen = value;
                }
            }
        }
        
        // Parse angleClose
        startPos = json.indexOf("\"angleClose\":");
        if (startPos != -1) {
            startPos += 12; // length of "angleClose":
            endPos = json.indexOf(",", startPos);
            if (endPos != -1) {
                String valueStr = json.substring(startPos, endPos);
                valueStr.trim();
                int value = valueStr.toInt();
                if (value >= 0 && value <= 180) {
                    newAngleClose = value;
                }
            }
        }
        
        // Parse anglePark
        startPos = json.indexOf("\"anglePark\":");
        if (startPos != -1) {
            startPos += 11; // length of "anglePark":
            endPos = json.indexOf("}", startPos);
            if (endPos != -1) {
                String valueStr = json.substring(startPos, endPos);
                valueStr.trim();
                int value = valueStr.toInt();
                if (value >= 0 && value <= 180) {
                    newAnglePark = value;
                }
            }
        }
        
        // If all values are valid, save them
        if (newAngleOpen >= 0 && newAngleOpen <= 180 &&
            newAngleClose >= 0 && newAngleClose <= 180 &&
            newAnglePark >= 0 && newAnglePark <= 180) {
            
            settings.angleOpen = newAngleOpen;
            settings.angleClose = newAngleClose;
            settings.anglePark = newAnglePark;
            
            saveSettingsToEEPROM();
            server.send(200, "text/plain", "OK");
        } else {
            server.send(400, "text/plain", "Invalid angle values");
        }
    } else {
        server.send(400, "text/plain", "No data received");
    }
}