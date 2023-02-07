# Импортирование необходимых модулей
import pygame
import os
# Инициализация параметров
pygame.mixer.init(96000, -16, 2, 8192)
pygame.mixer.music.set_volume(2.0)
HomeDir = os.path.expanduser("~")
MusicFormats = ['.mp3','.flac','.ogg','.aac','.wav','.aiff','.dsd','.mqa','.wma','.alac','.pcm']
ProjectDir = os.path.dirname(os.path.realpath(__file__))
MusicPath = HomeDir + r'\Music'
MusicFiles = []

class MusicManager:
    def __init__(self):
        self.MusicNumber = 0
        self.PausedMusic = False
        self.PlayingMusic = False
        self.StoppedMusic = False

    def StopMusic(self):
        pygame.mixer.music.stop()
        self.StoppedMusic = True
        self.PlayingMusic = False
        self.PausedMusic = False

    def PauseMusic(self):
        pygame.mixer.music.pause()
        self.PausedMusic = True
        self.PlayingMusic = False
        self.StoppedMusic = False

    def UnpauseMusic(self):
        pygame.mixer.music.unpause()
        self.PausedMusic = False
        self.PlayingMusic = True
        self.StoppedMusic = False

    def PlayMusic(self):
        self.PlayingMusic = True
        self.StoppedMusic = False
        self.PausedMusic = False
        for dir, subdir, files in os.walk(MusicPath):
            for file in files:
                # print(os.path.join(dir, file))
                file = os.path.normpath( os.path.join(dir, file))
                format = os.path.splitext(os.path.join(dir, file))[1]
                for MusicFormat in MusicFormats:
                    if format == MusicFormat:
                        MusicFiles.append(file)
        for Music in MusicFiles:
            pygame.mixer.music.load(Music)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pos = pygame.mixer.music.get_pos()/ 1000
            self.MusicNumber += 1

# if __name__ == "__main__":
#     manager = MusicManager()
#     thread = threading.Thread(target=manager.PlayMusic)
#     thread.start()
#     while True:
#         command = input(">>>")
#         print(command)
#         if command == 'pause':
#             pygame.mixer.music.pause()
#         elif command == 'unpause':
#             pygame.mixer.music.unpause()
#         elif command == 'stop':
#             pygame.mixer.music.stop()