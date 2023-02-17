import threading 
import time
import platform
from plyer import notification 

class Stopwatch:
    def __init__(self):
        self.TimeStarted = False
        self.TimeStopped = False
        self.TimeUnpaused = False
        self.TimePaused = False
        self.Time = 0
        self.TimeThread = None

    def Notification(self,title,message):
        notification.notify(  
            title = title,  
            message = message,  
            app_icon = None,  
            timeout = 10,  
            toast = False  
        ) 

    def Main(self):
        self.TimeThread = threading.Thread(target = self.Start).start()

    def Start(self):
        # self.Notification("Таймер","Таймер запущен")
        if self.TimeStarted == False  and self.TimeUnpaused == False and self.TimePaused == False:
            self.TimeStarted = True
            self.TimeStopped = False
            self.TimePaused = False
            self.TimeUnpaused = False
            self.LocalTime = 0

            while True:
                if self.TimeStopped == True:
                    break
                elif self.TimePaused:
                    continue
                else:
                    self.LocalTime += 1
                    time.sleep(1)
                    print(self.LocalTime)
            return True
        else:
            return False

    def Stop(self):
        if self.TimeStarted == True and self.TimeStopped == False:
            self.TimeStopped = True
            self.TimeStarted = True
            self.TimeUnpaused = False
            self.TimePaused = False
            return True
        else:
            return False

    def Pause(self):
        if self.TimeStarted == True and self.TimeStopped == False and self.TimePaused == False:
            self.TimePaused = True
            self.TimeStopped = False
            self.TimeStarted = True
            self.TimeUnpaused = False
            return True
        else:
            return False
    
    def Unpause(self):
        if self.TimeStarted == True and self.TimeStopped == False and self.TimeUnpaused == False and self.TimePaused == True:
            self.TimePaused = False
            self.TimeStarted = True
            self.TimeStopped = False
            self.TimeUnpaused = True
            return True
        else:
            return False

# if __name__ == "__main__":
#     timer = Stopwatch()
#     timer.Main()
#     time.sleep(5)
#     timer.Pause()
#     time.sleep(2)
#     timer.Unpause()
#     time.sleep(2)
    # timer.Stop()