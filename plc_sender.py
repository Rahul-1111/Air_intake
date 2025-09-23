import pymcprotocol

class PLCController:
    def __init__(self, ip, trigger_addr, result_addr):
        self.trigger_addr = trigger_addr
        self.result_addr = result_addr
        self.mc = pymcprotocol.Type3E()
        try:
            self.mc.connect(ip, 5007)
            print(f"✅ Connected to PLC at {ip}")
        except Exception as e:
            print(f"❌ PLC connection failed: {e}")

    def read_trigger(self):
        try:
            data = self.mc.batchread_wordunits(headdevice=f"D{self.trigger_addr}", readsize=1)
            return data[0] == 1
        except Exception as e:
            print(f"❌ Trigger read error: {e}")
            return False

    def reset_trigger(self):
        try:
            self.mc.batchwrite_wordunits(headdevice=f"D{self.trigger_addr}", values=[0])
            print("🔄 PLC Trigger Reset (D10=0)")
        except Exception as e:
            print(f"❌ Trigger reset failed: {e}")

    def write_result(self, value):
        try:
            self.mc.batchwrite_wordunits(headdevice=f"D{self.result_addr}", values=[value])
            print(f"📤 Wrote result to D{self.result_addr} = {value}")
        except Exception as e:
            print(f"❌ PLC write error: {e}")
