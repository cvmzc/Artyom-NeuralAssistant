import threading 
import time
import platform
if platform.system() == 'Windows':
    from win10toast import ToastNotifier

class Timer:
    def __init__(self):
        self.TimeStarted = False
        self.TimeStopped = False
        self.TimeUnpaused = False
        self.TimePaused = False
        self.Time = 0
        self.TimeThread = None

    def Notification(self,title,message):
        if platform.system() == 'Windows':
            ToastNotifier().show_toast(title=title,msg=message,duration=5)

    def Main(self,hours,minutes,seconds):
        self.TimeThread = threading.Thread(target = self.Start,args=(hours,minutes,seconds)).start()

    def Start(self,hours,minutes,seconds):
        # self.Notification("Таймер","Таймер запущен")
        if self.TimeStarted == False  and self.TimeUnpaused == False and self.TimePaused == False:
            self.TimeStarted = True
            self.TimeStopped = False
            self.TimePaused = False
            self.TimeUnpaused = False

            self.Time = int((hours * 3600) + (minutes * 60) + seconds)
            while self.Time != 0:
                if self.TimeStopped == True:
                    break
                elif self.TimePaused:
                    continue
                else:
                    self.Time -= 1
                    time.sleep(1)
                    print(self.Time)
            self.Notification("Таймер","Время вышло")
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
#     timer = Timer()
#     timer.Main(0,1,2)
    # time.sleep(5)
    # timer.Pause()
    # time.sleep(2)
    # timer.Unpause()
    # time.sleep(2)
    # timer.Stop()
