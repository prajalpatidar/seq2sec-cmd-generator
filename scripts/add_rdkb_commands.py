#!/usr/bin/env python3
"""Add RDKB (RDK Broadband) dmcli commands to the dataset."""

import json
import os

# RDKB dmcli commands based on RDKCENTRAL documentation
rdkb_commands = [
    # Device Info
    {"input": "get device model name", "output": "dmcli eRT getv Device.DeviceInfo.ModelName"},
    {"input": "show device model name", "output": "dmcli eRT getv Device.DeviceInfo.ModelName"},
    {"input": "display device manufacturer", "output": "dmcli eRT getv Device.DeviceInfo.Manufacturer"},
    {"input": "get device serial number", "output": "dmcli eRT getv Device.DeviceInfo.SerialNumber"},
    {"input": "show hardware version", "output": "dmcli eRT getv Device.DeviceInfo.HardwareVersion"},
    {"input": "display software version", "output": "dmcli eRT getv Device.DeviceInfo.SoftwareVersion"},
    {"input": "get firmware version", "output": "dmcli eRT getv Device.DeviceInfo.X_CISCO_COM_FirmwareName"},
    {"input": "show device uptime", "output": "dmcli eRT getv Device.DeviceInfo.UpTime"},
    {"input": "display device status", "output": "dmcli eRT getv Device.DeviceInfo.X_CISCO_COM_BootTime"},
    {"input": "get mac address", "output": "dmcli eRT getv Device.DeviceInfo.X_COMCAST-COM_CM_MAC"},
    {"input": "show wan mac address", "output": "dmcli eRT getv Device.DeviceInfo.X_COMCAST-COM_WAN_MAC"},
    {"input": "retrieve device description", "output": "dmcli eRT getv Device.DeviceInfo.Description"},
    
    # WiFi Settings
    {"input": "get wifi ssid", "output": "dmcli eRT getv Device.WiFi.SSID.1.SSID"},
    {"input": "show wifi ssid", "output": "dmcli eRT getv Device.WiFi.SSID.1.SSID"},
    {"input": "display wifi password", "output": "dmcli eRT getv Device.WiFi.AccessPoint.1.Security.X_COMCAST-COM_KeyPassphrase"},
    {"input": "get wifi encryption mode", "output": "dmcli eRT getv Device.WiFi.AccessPoint.1.Security.ModeEnabled"},
    {"input": "show wifi channel", "output": "dmcli eRT getv Device.WiFi.Radio.1.Channel"},
    {"input": "display wifi status", "output": "dmcli eRT getv Device.WiFi.Radio.1.Enable"},
    {"input": "get wifi frequency band", "output": "dmcli eRT getv Device.WiFi.Radio.1.OperatingFrequencyBand"},
    {"input": "show wifi radio enable", "output": "dmcli eRT getv Device.WiFi.Radio.1.Enable"},
    {"input": "display connected wifi clients", "output": "dmcli eRT getv Device.WiFi.AccessPoint.1.AssociatedDeviceNumberOfEntries"},
    {"input": "get wifi bandwidth", "output": "dmcli eRT getv Device.WiFi.Radio.1.OperatingChannelBandwidth"},
    {"input": "show wifi transmit power", "output": "dmcli eRT getv Device.WiFi.Radio.1.TransmitPower"},
    {"input": "set wifi ssid", "output": "dmcli eRT setv Device.WiFi.SSID.1.SSID string"},
    {"input": "enable wifi radio", "output": "dmcli eRT setv Device.WiFi.Radio.1.Enable bool true"},
    {"input": "disable wifi radio", "output": "dmcli eRT setv Device.WiFi.Radio.1.Enable bool false"},
    {"input": "set wifi channel", "output": "dmcli eRT setv Device.WiFi.Radio.1.Channel uint"},
    {"input": "change wifi password", "output": "dmcli eRT setv Device.WiFi.AccessPoint.1.Security.X_COMCAST-COM_KeyPassphrase string"},
    
    # LAN Settings
    {"input": "get lan ip address", "output": "dmcli eRT getv Device.X_CISCO_COM_DeviceControl.LanManagementEntry.1.LanIPAddress"},
    {"input": "show lan subnet mask", "output": "dmcli eRT getv Device.X_CISCO_COM_DeviceControl.LanManagementEntry.1.LanSubnetMask"},
    {"input": "display dhcp server enable", "output": "dmcli eRT getv Device.DHCPv4.Server.Enable"},
    {"input": "get dhcp start address", "output": "dmcli eRT getv Device.DHCPv4.Server.Pool.1.MinAddress"},
    {"input": "show dhcp end address", "output": "dmcli eRT getv Device.DHCPv4.Server.Pool.1.MaxAddress"},
    {"input": "display dhcp lease time", "output": "dmcli eRT getv Device.DHCPv4.Server.Pool.1.LeaseTime"},
    {"input": "get dhcp clients", "output": "dmcli eRT getv Device.DHCPv4.Server.Pool.1.Client."},
    {"input": "set lan ip address", "output": "dmcli eRT setv Device.X_CISCO_COM_DeviceControl.LanManagementEntry.1.LanIPAddress string"},
    {"input": "enable dhcp server", "output": "dmcli eRT setv Device.DHCPv4.Server.Enable bool true"},
    {"input": "disable dhcp server", "output": "dmcli eRT setv Device.DHCPv4.Server.Enable bool false"},
    
    # WAN Settings
    {"input": "get wan ip address", "output": "dmcli eRT getv Device.X_CISCO_COM_DeviceControl.WanAddressMode"},
    {"input": "show wan connection status", "output": "dmcli eRT getv Device.X_CISCO_COM_DeviceControl.WanCurrentState"},
    {"input": "display wan ipv6 address", "output": "dmcli eRT getv Device.IP.Interface.1.IPv6Address.1.IPAddress"},
    {"input": "get wan link status", "output": "dmcli eRT getv Device.Ethernet.Link.1.Status"},
    {"input": "show wan mac address", "output": "dmcli eRT getv Device.Ethernet.Link.1.MACAddress"},
    
    # NAT and Firewall
    {"input": "get nat enable status", "output": "dmcli eRT getv Device.NAT.X_CISCO_COM_EnableNATMapping"},
    {"input": "show port forwarding rules", "output": "dmcli eRT getv Device.NAT.PortMapping."},
    {"input": "display firewall status", "output": "dmcli eRT getv Device.X_CISCO_COM_Security.Firewall.FirewallLevel"},
    {"input": "get dmz host", "output": "dmcli eRT getv Device.NAT.X_CISCO_COM_DMZ.Enable"},
    {"input": "add port forwarding rule", "output": "dmcli eRT setv Device.NAT.PortMapping.1.Enable bool true"},
    {"input": "enable nat", "output": "dmcli eRT setv Device.NAT.X_CISCO_COM_EnableNATMapping bool true"},
    {"input": "set firewall level", "output": "dmcli eRT setv Device.X_CISCO_COM_Security.Firewall.FirewallLevel string"},
    
    # Parental Control
    {"input": "get parental control status", "output": "dmcli eRT getv Device.X_Comcast_com_ParentalControl.ManagedSites.Enable"},
    {"input": "show blocked sites", "output": "dmcli eRT getv Device.X_Comcast_com_ParentalControl.ManagedSites.BlockedSite."},
    {"input": "display parental control services", "output": "dmcli eRT getv Device.X_Comcast_com_ParentalControl.ManagedServices.Service."},
    {"input": "enable parental control", "output": "dmcli eRT setv Device.X_Comcast_com_ParentalControl.ManagedSites.Enable bool true"},
    
    # Time Settings
    {"input": "get current time", "output": "dmcli eRT getv Device.Time.CurrentLocalTime"},
    {"input": "show time zone", "output": "dmcli eRT getv Device.Time.LocalTimeZone"},
    {"input": "display ntp server", "output": "dmcli eRT getv Device.Time.NTPServer1"},
    {"input": "set time zone", "output": "dmcli eRT setv Device.Time.LocalTimeZone string"},
    
    # Logging and Diagnostics
    {"input": "get log entries", "output": "dmcli eRT getv Device.DeviceInfo.X_RDKCENTRAL-COM_xOpsDeviceMgmt.Logging."},
    {"input": "show system log", "output": "dmcli eRT getv Device.DeviceInfo.X_RDKCENTRAL-COM_xOpsDeviceMgmt.Logging.xOpsDMUploadLogsNow"},
    {"input": "display memory usage", "output": "dmcli eRT getv Device.DeviceInfo.MemoryStatus.Total"},
    {"input": "get cpu usage", "output": "dmcli eRT getv Device.DeviceInfo.ProcessStatus.CPUUsage"},
    {"input": "show process status", "output": "dmcli eRT getv Device.DeviceInfo.ProcessStatus.Process."},
    {"input": "trigger log upload", "output": "dmcli eRT setv Device.DeviceInfo.X_RDKCENTRAL-COM_xOpsDeviceMgmt.Logging.xOpsDMUploadLogsNow bool true"},
    
    # TR-069 Management
    {"input": "get acs url", "output": "dmcli eRT getv Device.ManagementServer.URL"},
    {"input": "show acs connection status", "output": "dmcli eRT getv Device.ManagementServer.X_CISCO_COM_ConnectionRequestURLPort"},
    {"input": "display periodic inform enable", "output": "dmcli eRT getv Device.ManagementServer.PeriodicInformEnable"},
    {"input": "get periodic inform interval", "output": "dmcli eRT getv Device.ManagementServer.PeriodicInformInterval"},
    {"input": "set acs url", "output": "dmcli eRT setv Device.ManagementServer.URL string"},
    {"input": "enable periodic inform", "output": "dmcli eRT setv Device.ManagementServer.PeriodicInformEnable bool true"},
    
    # Cable Modem Status
    {"input": "get cm status", "output": "dmcli eRT getv Device.X_CISCO_COM_CableModem.DOCSISVersion"},
    {"input": "show docsis version", "output": "dmcli eRT getv Device.X_CISCO_COM_CableModem.DOCSISVersion"},
    {"input": "display cm downstream power", "output": "dmcli eRT getv Device.X_CISCO_COM_CableModem.DownstreamPowerLevel"},
    {"input": "get cm upstream power", "output": "dmcli eRT getv Device.X_CISCO_COM_CableModem.UpstreamPowerLevel"},
    {"input": "show cm snr", "output": "dmcli eRT getv Device.X_CISCO_COM_CableModem.SNRLevel"},
    {"input": "display cm channels", "output": "dmcli eRT getv Device.X_CISCO_COM_CableModem.DOCSISDownstreamChannel."},
    
    # MoCA Settings
    {"input": "get moca status", "output": "dmcli eRT getv Device.MoCA.Interface.1.Enable"},
    {"input": "show moca frequency", "output": "dmcli eRT getv Device.MoCA.Interface.1.FreqCurrentMask"},
    {"input": "display moca nodes", "output": "dmcli eRT getv Device.MoCA.Interface.1.AssociatedDevice."},
    {"input": "enable moca", "output": "dmcli eRT setv Device.MoCA.Interface.1.Enable bool true"},
    {"input": "disable moca", "output": "dmcli eRT setv Device.MoCA.Interface.1.Enable bool false"},
    
    # Ethernet Port Status
    {"input": "get ethernet port status", "output": "dmcli eRT getv Device.Ethernet.Interface.1.Status"},
    {"input": "show ethernet link speed", "output": "dmcli eRT getv Device.Ethernet.Interface.1.CurrentBitRate"},
    {"input": "display ethernet duplex mode", "output": "dmcli eRT getv Device.Ethernet.Interface.1.DuplexMode"},
    
    # USB Settings
    {"input": "get usb status", "output": "dmcli eRT getv Device.USB.Interface.1.Enable"},
    {"input": "show usb devices", "output": "dmcli eRT getv Device.USB.USBHosts.Host.1.Device."},
    
    # Remote Management
    {"input": "get remote access enable", "output": "dmcli eRT getv Device.UserInterface.X_CISCO_COM_RemoteAccess.Enable"},
    {"input": "show remote access port", "output": "dmcli eRT getv Device.UserInterface.X_CISCO_COM_RemoteAccess.Port"},
    {"input": "enable remote access", "output": "dmcli eRT setv Device.UserInterface.X_CISCO_COM_RemoteAccess.Enable bool true"},
    
    # Factory Reset
    {"input": "factory reset device", "output": "dmcli eRT setv Device.X_CISCO_COM_DeviceControl.FactoryReset string Router,Wifi,VoIP,Dect,MoCA"},
    {"input": "reboot device", "output": "dmcli eRT setv Device.X_CISCO_COM_DeviceControl.RebootDevice string Device"},
    {"input": "reset wifi settings", "output": "dmcli eRT setv Device.X_CISCO_COM_DeviceControl.FactoryReset string Wifi"},
    
    # QoS Settings
    {"input": "get qos enable", "output": "dmcli eRT getv Device.X_CISCO_COM_DeviceControl.QoSEnable"},
    {"input": "enable qos", "output": "dmcli eRT setv Device.X_CISCO_COM_DeviceControl.QoSEnable bool true"},
    
    # Bridge Mode
    {"input": "get bridge mode status", "output": "dmcli eRT getv Device.X_CISCO_COM_DeviceControl.LanMode"},
    {"input": "enable bridge mode", "output": "dmcli eRT setv Device.X_CISCO_COM_DeviceControl.LanMode string bridge-static"},
    {"input": "disable bridge mode", "output": "dmcli eRT setv Device.X_CISCO_COM_DeviceControl.LanMode string router"},
    
    # Advanced WiFi
    {"input": "get wifi country code", "output": "dmcli eRT getv Device.WiFi.Radio.1.RegulatoryDomain"},
    {"input": "show wifi operating standard", "output": "dmcli eRT getv Device.WiFi.Radio.1.OperatingStandards"},
    {"input": "display wifi beacon interval", "output": "dmcli eRT getv Device.WiFi.Radio.1.BeaconPeriod"},
    {"input": "get wifi dtim interval", "output": "dmcli eRT getv Device.WiFi.Radio.1.DTIMPeriod"},
    {"input": "show wifi guard interval", "output": "dmcli eRT getv Device.WiFi.Radio.1.GuardInterval"},
    {"input": "display wifi wmm enable", "output": "dmcli eRT getv Device.WiFi.AccessPoint.1.WMMEnable"},
    {"input": "get wifi mac filter mode", "output": "dmcli eRT getv Device.WiFi.AccessPoint.1.MACAddressControlEnabled"},
    
    # Guest WiFi
    {"input": "get guest wifi ssid", "output": "dmcli eRT getv Device.WiFi.SSID.5.SSID"},
    {"input": "show guest wifi status", "output": "dmcli eRT getv Device.WiFi.SSID.5.Enable"},
    {"input": "enable guest wifi", "output": "dmcli eRT setv Device.WiFi.SSID.5.Enable bool true"},
    {"input": "disable guest wifi", "output": "dmcli eRT setv Device.WiFi.SSID.5.Enable bool false"},
    {"input": "set guest wifi ssid", "output": "dmcli eRT setv Device.WiFi.SSID.5.SSID string"},
    
    # IPv6
    {"input": "get ipv6 enable", "output": "dmcli eRT getv Device.IP.IPv6Enable"},
    {"input": "show ipv6 address", "output": "dmcli eRT getv Device.IP.Interface.1.IPv6Address."},
    {"input": "enable ipv6", "output": "dmcli eRT setv Device.IP.IPv6Enable bool true"},
    
    # DNS Settings
    {"input": "get dns servers", "output": "dmcli eRT getv Device.DNS.Client.Server."},
    {"input": "show primary dns", "output": "dmcli eRT getv Device.DNS.Client.Server.1.DNSServer"},
    {"input": "set dns server", "output": "dmcli eRT setv Device.DNS.Client.Server.1.DNSServer string"},
]

def add_rdkb_commands(dataset_path):
    """Add RDKB commands to existing dataset."""
    
    # Load existing dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    original_size = len(data)
    print(f"Original dataset size: {original_size}")
    
    # Add RDKB commands
    added = 0
    for entry in rdkb_commands:
        # Check if entry already exists
        if not any(d['input'] == entry['input'] for d in data):
            data.append(entry)
            added += 1
    
    # Save updated dataset
    with open(dataset_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Added {added} RDKB dmcli commands")
    print(f"New dataset size: {len(data)}")
    print(f"Saved to: {dataset_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "data", "commands-dataset.json")
    add_rdkb_commands(dataset_path)
