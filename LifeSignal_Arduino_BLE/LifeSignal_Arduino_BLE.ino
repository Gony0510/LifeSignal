#include <Arduino.h>
#include <HardwareSerial.h>
#include <Preferences.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <BLE2902.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include "DFRobot_C4001.h"

// LifeSignal C4001 + BLE provisioning firmware.
// The original LifeSignal_Arduino/LifeSignal_Arduino.ino is intentionally kept unchanged.

const int ROOM_NUMBER = 402;
const String ZONE_LOCATION = "방B";
const char *SENSOR_NAME = "C4001";

const int DEFAULT_SERVER_PORT = 8881;
const unsigned long WIFI_CONNECT_TIMEOUT_MS = 20000;
const unsigned long SAMPLE_INTERVAL_MS = 100;
const unsigned long SEND_INTERVAL_MS = 1010;
const uint32_t TARGET_ENERGY_THRESHOLD = 5000;
const char *CAPTURE_START_COMMAND = "C4001_CAPTURE_START";
const char *CAPTURE_STOP_COMMAND = "C4001_CAPTURE_STOP";
const char *CAPTURE_SAMPLE_PREFIX = "@C4001_SAMPLE ";

// LifeSignal provisioning BLE service and characteristics.
// Keep these UUIDs synchronized with BLEProvisioningManager.swift.
const char *SERVICE_UUID = "7b6f0001-6d5f-4c20-9f4b-2e1d7a0c1000";
const char *SSID_UUID = "7b6f0002-6d5f-4c20-9f4b-2e1d7a0c1000";
const char *PASSWORD_UUID = "7b6f0003-6d5f-4c20-9f4b-2e1d7a0c1000";
const char *SERVER_HOST_UUID = "7b6f0004-6d5f-4c20-9f4b-2e1d7a0c1000";
const char *SERVER_PORT_UUID = "7b6f0005-6d5f-4c20-9f4b-2e1d7a0c1000";
const char *COMMAND_UUID = "7b6f0006-6d5f-4c20-9f4b-2e1d7a0c1000";
const char *STATUS_UUID = "7b6f0007-6d5f-4c20-9f4b-2e1d7a0c1000";

HardwareSerial mySerial(1);
DFRobot_C4001_UART sensor(&mySerial, 9600, /*rx*/ D2, /*tx*/ D3);
WebSocketsClient webSocket;
Preferences preferences;

BLEServer *bleServer = nullptr;
BLECharacteristic *statusCharacteristic = nullptr;

String wifiSSID;
String wifiPassword;
String serverHost;
uint16_t serverPort = DEFAULT_SERVER_PORT;

String draftSSID;
String draftPassword;
String draftServerHost;
uint16_t draftServerPort = DEFAULT_SERVER_PORT;

String latestStatusJSON;
bool bleClientConnected = false;
bool pendingConnectCommand = false;
bool pendingClearCommand = false;
bool pendingStatusCommand = false;
bool wifiConnecting = false;
bool webSocketConfigured = false;
bool webSocketConnected = false;
bool sensorReady = false;
bool serialCaptureEnabled = false;
unsigned long wifiConnectStartedAt = 0;
String serialCommandBuffer;

const int ledPin = 13;
unsigned long lastSendTime = 0;
unsigned long lastSampleTime = 0;
int detectCount = 0;
int noDetectCount = 0;
bool finalPresence = false;
uint32_t latestTargetEnergy = 0;

struct C4001Sample {
  unsigned long sampleMillis;
  bool motion;
  bool instantPresence;
  uint8_t targetNumber;
  float targetSpeedMps;
  float targetRangeM;
  uint32_t targetEnergy;
};

String jsonEscape(const String &value);
void publishStatus(const String &state, const String &message);
void beginProvisionedConnection();
void clearStoredConfiguration();
void processProvisioningCommands();
void processSerialCaptureCommands();
void emitC4001Sample(const C4001Sample &sample);

enum class ProvisioningField {
  SSID,
  PASSWORD,
  SERVER_HOST,
  SERVER_PORT,
  COMMAND
};

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override {
    bleClientConnected = true;
    publishStatus(
      WiFi.status() == WL_CONNECTED ? (webSocketConnected ? "ready" : "wifi_connected") : "ble_ready",
      "BLE client connected"
    );
  }

  void onDisconnect(BLEServer *server) override {
    bleClientConnected = false;
    BLEDevice::startAdvertising();
  }
};

class FieldWriteCallbacks : public BLECharacteristicCallbacks {
 public:
  explicit FieldWriteCallbacks(ProvisioningField field) : field_(field) {}

  void onWrite(BLECharacteristic *characteristic) override {
    String value = characteristic->getValue().c_str();

    switch (field_) {
      case ProvisioningField::SSID:
        draftSSID = value.substring(0, 32);
        break;
      case ProvisioningField::PASSWORD:
        draftPassword = value.substring(0, 63);
        break;
      case ProvisioningField::SERVER_HOST:
        draftServerHost = value.substring(0, 64);
        break;
      case ProvisioningField::SERVER_PORT: {
        long parsedPort = value.toInt();
        if (parsedPort > 0 && parsedPort <= 65535) {
          draftServerPort = static_cast<uint16_t>(parsedPort);
        }
        break;
      }
      case ProvisioningField::COMMAND:
        value.trim();
        value.toLowerCase();
        if (value == "connect") {
          pendingConnectCommand = true;
        } else if (value == "clear") {
          pendingClearCommand = true;
        } else if (value == "status") {
          pendingStatusCommand = true;
        }
        break;
    }
  }

 private:
  ProvisioningField field_;
};

BLECharacteristic *makeWriteCharacteristic(
  BLEService *service,
  const char *uuid,
  ProvisioningField field
) {
  BLECharacteristic *characteristic = service->createCharacteristic(
    uuid,
    BLECharacteristic::PROPERTY_WRITE
  );
  characteristic->setCallbacks(new FieldWriteCallbacks(field));
  return characteristic;
}

String jsonEscape(const String &value) {
  String escaped;
  escaped.reserve(value.length() + 8);
  for (size_t index = 0; index < value.length(); index++) {
    char character = value.charAt(index);
    if (character == '\\' || character == '"') {
      escaped += '\\';
      escaped += character;
    } else if (character == '\n') {
      escaped += "\\n";
    } else if (character != '\r') {
      escaped += character;
    }
  }
  return escaped;
}

void publishStatus(const String &state, const String &message) {
  String deviceIP = WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString() : "";
  String endpoint = serverHost.length() > 0
    ? serverHost + ":" + String(serverPort)
    : "";

  latestStatusJSON =
    "{\"state\":\"" + jsonEscape(state) +
    "\",\"message\":\"" + jsonEscape(message) +
    "\",\"deviceIP\":\"" + jsonEscape(deviceIP) +
    "\",\"serverEndpoint\":\"" + jsonEscape(endpoint) + "\"}";

  Serial.println("BLE 상태: " + latestStatusJSON);

  if (statusCharacteristic != nullptr) {
    statusCharacteristic->setValue(latestStatusJSON.c_str());
    if (bleClientConnected) {
      statusCharacteristic->notify();
    }
  }
}

void setupBLE() {
  uint64_t chipID = ESP.getEfuseMac();
  char deviceName[24];
  snprintf(
    deviceName,
    sizeof(deviceName),
    "LifeSignal-%04X",
    static_cast<uint16_t>(chipID & 0xFFFF)
  );

  BLEDevice::init(deviceName);
  bleServer = BLEDevice::createServer();
  bleServer->setCallbacks(new ServerCallbacks());

  BLEService *service = bleServer->createService(SERVICE_UUID);
  makeWriteCharacteristic(service, SSID_UUID, ProvisioningField::SSID);
  makeWriteCharacteristic(service, PASSWORD_UUID, ProvisioningField::PASSWORD);
  makeWriteCharacteristic(service, SERVER_HOST_UUID, ProvisioningField::SERVER_HOST);
  makeWriteCharacteristic(service, SERVER_PORT_UUID, ProvisioningField::SERVER_PORT);
  makeWriteCharacteristic(service, COMMAND_UUID, ProvisioningField::COMMAND);

  statusCharacteristic = service->createCharacteristic(
    STATUS_UUID,
    BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
  );
  statusCharacteristic->addDescriptor(new BLE2902());

  service->start();

  BLEAdvertising *advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setScanResponse(true);
  advertising->setMinPreferred(0x06);
  advertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();

  publishStatus("ble_ready", "Waiting for provisioning");
  Serial.println("BLE 프로비저닝 대기: " + String(deviceName));
}

bool hasCompleteConfiguration() {
  return wifiSSID.length() > 0 && serverHost.length() > 0 && serverPort > 0;
}

void loadStoredConfiguration() {
  wifiSSID = preferences.getString("ssid", "");
  wifiPassword = preferences.getString("password", "");
  serverHost = preferences.getString("server", "");
  serverPort = preferences.getUShort("port", DEFAULT_SERVER_PORT);

  draftSSID = wifiSSID;
  draftPassword = wifiPassword;
  draftServerHost = serverHost;
  draftServerPort = serverPort;
}

void saveDraftConfiguration() {
  wifiSSID = draftSSID;
  wifiPassword = draftPassword;
  serverHost = draftServerHost;
  serverPort = draftServerPort;

  preferences.putString("ssid", wifiSSID);
  preferences.putString("password", wifiPassword);
  preferences.putString("server", serverHost);
  preferences.putUShort("port", serverPort);
}

void beginProvisionedConnection() {
  if (!hasCompleteConfiguration()) {
    publishStatus("missing_config", "SSID and server address are required");
    return;
  }

  if (webSocketConfigured) {
    webSocketConfigured = false;
    webSocketConnected = false;
    webSocket.disconnect();
  }

  WiFi.disconnect();
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(wifiSSID.c_str(), wifiPassword.c_str());
  wifiConnecting = true;
  wifiConnectStartedAt = millis();

  publishStatus("wifi_connecting", "Connecting to Wi-Fi");
  Serial.println("Wi-Fi 연결 시도: " + wifiSSID);
}

void clearStoredConfiguration() {
  preferences.clear();
  wifiSSID = "";
  wifiPassword = "";
  serverHost = "";
  serverPort = DEFAULT_SERVER_PORT;
  draftSSID = "";
  draftPassword = "";
  draftServerHost = "";
  draftServerPort = DEFAULT_SERVER_PORT;

  if (webSocketConfigured) {
    webSocketConfigured = false;
    webSocket.disconnect();
  }
  webSocketConnected = false;
  wifiConnecting = false;
  WiFi.setAutoReconnect(false);
  WiFi.disconnect();
  publishStatus("config_cleared", "Stored configuration cleared");
}

void processProvisioningCommands() {
  if (pendingClearCommand) {
    pendingClearCommand = false;
    clearStoredConfiguration();
  }

  if (pendingConnectCommand) {
    pendingConnectCommand = false;

    if (draftSSID.length() == 0 || draftServerHost.length() == 0) {
      publishStatus("missing_config", "SSID and server address are required");
    } else {
      saveDraftConfiguration();
      publishStatus("config_saved", "Configuration saved");
      beginProvisionedConnection();
    }
  }

  if (pendingStatusCommand) {
    pendingStatusCommand = false;
    publishStatus(
      WiFi.status() == WL_CONNECTED ? (webSocketConnected ? "ready" : "wifi_connected") : "ble_ready",
      "Status refreshed"
    );
  }
}

void webSocketEvent(WStype_t type, uint8_t *payload, size_t length) {
  if (type == WStype_CONNECTED) {
    webSocketConnected = true;
    publishStatus("ready", "Wi-Fi and server connected");
    Serial.println("🟢 중앙 서버에 접속 성공!");
  } else if (type == WStype_DISCONNECTED) {
    bool wasConnected = webSocketConnected;
    webSocketConnected = false;
    if (webSocketConfigured && WiFi.status() == WL_CONNECTED && wasConnected) {
      publishStatus("server_disconnected", "Server disconnected; retrying");
    }
    Serial.println("🔴 중앙 서버와 연결 끊김.");
  }
}

void updateNetworkState() {
  if (!wifiConnecting) {
    return;
  }

  if (WiFi.status() == WL_CONNECTED) {
    wifiConnecting = false;
    publishStatus("wifi_connected", "Wi-Fi connected");
    Serial.println("🟢 Wi-Fi 연결 성공: " + WiFi.localIP().toString());

    webSocket.begin(serverHost.c_str(), serverPort, "/");
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(5000);
    webSocketConfigured = true;
    publishStatus("server_connecting", "Connecting to monitoring server");
    return;
  }

  if (millis() - wifiConnectStartedAt >= WIFI_CONNECT_TIMEOUT_MS) {
    wifiConnecting = false;
    WiFi.disconnect();
    publishStatus("wifi_failed", "Wi-Fi connection timed out");
    Serial.println("❌ Wi-Fi 연결 시간 초과");
  }
}

void setupSensor() {
  if (!sensor.begin()) {
    Serial.println("❌ 센서 연결 실패!");
    sensorReady = false;
    return;
  }

  sensorReady = true;
  Serial.println("🟢 센서 연결 성공!");
  sensor.setSensorMode(eSpeedMode);
  sensor.setDetectionRange(30, 60, 50);
  sensor.setTrigSensitivity(5);
  sensor.setKeepSensitivity(4);
}

void processSerialCaptureCommands() {
  while (Serial.available() > 0) {
    const char character = static_cast<char>(Serial.read());
    if (character == '\r') {
      continue;
    }
    if (character != '\n') {
      if (serialCommandBuffer.length() < 64) {
        serialCommandBuffer += character;
      }
      continue;
    }

    serialCommandBuffer.trim();
    if (serialCommandBuffer == CAPTURE_START_COMMAND) {
      serialCaptureEnabled = true;
    } else if (serialCommandBuffer == CAPTURE_STOP_COMMAND) {
      serialCaptureEnabled = false;
    }
    serialCommandBuffer = "";
  }
}

void emitC4001Sample(const C4001Sample &sample) {
  if (!serialCaptureEnabled) {
    return;
  }

  String samplePayload;
  samplePayload.reserve(480);
  samplePayload = "{\"type\":\"c4001_sample\"";
  samplePayload += ",\"sensor\":\"";
  samplePayload += SENSOR_NAME;
  samplePayload += "\",\"room\":";
  samplePayload += String(ROOM_NUMBER);
  samplePayload += ",\"location\":\"";
  samplePayload += jsonEscape(ZONE_LOCATION);
  samplePayload += "\",\"sample_millis\":";
  samplePayload += String(sample.sampleMillis);
  samplePayload += ",\"motion\":";
  samplePayload += sample.motion ? "true" : "false";
  samplePayload += ",\"instant_presence\":";
  samplePayload += sample.instantPresence ? "true" : "false";
  samplePayload += ",\"status\":";
  samplePayload += finalPresence ? "true" : "false";
  samplePayload += ",\"target_number\":";
  samplePayload += String(sample.targetNumber);
  samplePayload += ",\"target_speed_m_s\":";
  samplePayload += String(sample.targetSpeedMps, 3);
  samplePayload += ",\"target_range_m\":";
  samplePayload += String(sample.targetRangeM, 3);
  samplePayload += ",\"target_energy\":";
  samplePayload += String(sample.targetEnergy);
  samplePayload += "}";

  // 수집 프로그램만 이 접두사를 처리하며 일반 상태 출력에서는 숨깁니다.
  Serial.print(CAPTURE_SAMPLE_PREFIX);
  Serial.println(samplePayload);
}

void sampleSensor() {
  if (!sensorReady) {
    return;
  }

  const bool motionDetected = sensor.motionDetection();

  if (millis() - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = millis();

    C4001Sample sample;
    sample.sampleMillis = lastSampleTime;
    sample.targetNumber = sensor.getTargetNumber();
    sample.targetEnergy = sensor.getTargetEnergy();
    latestTargetEnergy = sample.targetEnergy;
    sample.targetSpeedMps = sensor.getTargetSpeed();
    sample.targetRangeM = sensor.getTargetRange();
    sample.motion = motionDetected;
    sample.instantPresence = sample.targetNumber > 0 &&
      sample.targetEnergy > TARGET_ENERGY_THRESHOLD;

    if (sample.instantPresence) {
      detectCount++;
    } else {
      noDetectCount++;
    }

    emitC4001Sample(sample);
  }
}

void sendRadarDataIfDue() {
  if (!sensorReady) {
    return;
  }

  if (millis() - lastSendTime < SEND_INTERVAL_MS) {
    return;
  }
  lastSendTime = millis();

  int total = detectCount + noDetectCount;
  if (total > 0) {
    finalPresence = static_cast<float>(detectCount) / total >= 0.70f;
  }

  String jsonPayload = "{\"type\":\"radar_data\"";
  jsonPayload += ",\"sensor\":\"";
  jsonPayload += SENSOR_NAME;
  jsonPayload += "\",\"room\":";
  jsonPayload += String(ROOM_NUMBER);
  jsonPayload += ",\"status\":";
  jsonPayload += finalPresence ? "true" : "false";
  jsonPayload += ",\"target_energy\":";
  jsonPayload += String(latestTargetEnergy);
  jsonPayload += ",\"location\":\"";
  jsonPayload += jsonEscape(ZONE_LOCATION);
  jsonPayload += "\"}";

  Serial.println("전송 데이터: " + jsonPayload);
  
  if (webSocketConnected) {
    webSocket.sendTXT(jsonPayload);
  }

  digitalWrite(ledPin, finalPresence ? HIGH : LOW);
  detectCount = 0;
  noDetectCount = 0;
}

void setup() {
  Serial.begin(115200);
  pinMode(ledPin, OUTPUT);
  delay(1000);

  preferences.begin("lifesignal", false);
  loadStoredConfiguration();
  setupBLE();
  setupSensor();

  if (hasCompleteConfiguration()) {
    beginProvisionedConnection();
  } else {
    publishStatus("missing_config", "Open the iPhone app to configure Wi-Fi");
  }
}

void loop() {
  processSerialCaptureCommands();
  processProvisioningCommands();
  updateNetworkState();

  if (webSocketConfigured) {
    webSocket.loop();
  }

  sampleSensor();
  sendRadarDataIfDue();
}
