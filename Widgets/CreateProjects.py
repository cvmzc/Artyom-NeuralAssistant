import sys
from PyQt5.QtWidgets import QApplication, QWidget
from UI.MainUI import Ui_WidgetCreateProject
from PyQt5 import QtWidgets
from PyQt5.QtGui import*
import os
from loguru import logger
import platform
from win10toast import ToastNotifier
from threading import Thread
import json

# Инициализация параметров для логирования
ProjectDir = os.path.dirname(os.path.realpath(__file__))
logger.add(os.path.join(ProjectDir,'Logs/CreateProjects.log'),format="{time} {level} {message}",level="INFO",rotation="200 MB",diagnose=True)

class CreateProject(QWidget):
    def __init__(self,type:str = None):
        super().__init__()
        self.TypeProject = type
        self.TypeFunctions = {"c++_project":self.CppProject,"django_project":self.DjangoProject,"python_project":self.PythonProject,"NodeJS_project":self.NodeJSProject,
                            "rust_project":self.RustProject,"c_project":self.CProject,"go_project":self.GoProject,"java_project":self.JavaProject
        }
        self.UI = Ui_WidgetCreateProject()
        self.UI.setupUi(self)
        self.Directory = None
        self.NameProject = None
        self.NameScript = None
        self.IconPath = os.path.dirname(os.path.realpath(__file__)) +"/Images/CreateProject.png"
        self.UI.ErrorNameProject.setHidden(True)
        self.UI.ErrorMainScript.setHidden(True)
        self.Interface()
        # self.setFixedSize(395,300)
        self.setWindowIcon(QIcon(self.IconPath))
        self.show()
    
    def OpenDirectory(self):
        # self.Directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Выберите папку для проекта:', r'C:\\')
        self.Directory = r"C:\Users\Blackflame576\Documents\Blackflame576\DigitalBit\Artyom-NeuralAssistant\Widgets"

    def Notification(self,title,message):
        if platform.system() == 'Windows':
            ToastNotifier().show_toast(title=title,msg=message,duration=5)

    def Interface(self):
        self.UI.DirectoryButton.clicked.connect(self.OpenDirectory)
        self.UI.CreateButton.clicked.connect(self.Create)
        if self.TypeProject == "python":
            self.UI.EnvCheckBox.setVisible(True)
            self.UI.EnvCheckBox.setChecked(True)
        elif self.TypeProject == "django":
            self.UI.NameMainScript.setEnabled(False)
            self.UI.EnvCheckBox.setVisible(True)
            self.UI.EnvCheckBox.setChecked(True)
        else:
            self.UI.EnvCheckBox.setVisible(False)
        self.UI.DockerCheckBox.setVisible(True)
        self.UI.DockerCheckBox.setChecked(True)
        self.UI.ProgressBar.setValue(0)
        self.UI.ProgressBar.setVisible(False)

    def JavaProject(self):
        print("Java")
        try:
            if self.NameScript.endswith(".java") == True:
                self.NameScript = self.NameScript.replace(".java","")
            os.chdir(self.Directory)
            os.mkdir(self.NameProject)
            os.chdir(os.path.join(self.Directory,self.NameProject))
            if not os.path.exists(self.NameProject):
                self.UI.ProgressBar.setVisible(True)
                self.UI.ProgressBar.setValue(5)
                GitIgnoreFile = open(os.path.join(ProjectDir,"Templates/GitIgnore/Java.gitignore"),"r")
                GitIgnoreData = GitIgnoreFile.read()
                GitIgnoreFile.close()
                GitIgnoreFile = open(".gitignore","w")
                GitIgnoreFile.write(GitIgnoreData)
                GitIgnoreFile.close()
                self.UI.ProgressBar.setValue(20)
                ScriptFile = open(os.path.join(ProjectDir,"Templates/JavaTemplate.java"),"r")
                ScriptFileData = ScriptFile.read()
                ScriptFile.close()
                ScriptFile = open(f"{self.NameScript}.java","w")
                ScriptFile.write(ScriptFileData)
                ScriptFile.close()
                self.UI.ProgressBar.setValue(50)
                if self.UI.DockerCheckBox.isChecked() == True:
                    DockergIgnoreFile = open(os.path.join(ProjectDir,"Templates/.dockerignore"),"r")
                    DockergIgnoreData = DockergIgnoreFile.read()
                    DockergIgnoreFile.close()
                    DockergIgnoreFile = open(".dockerignore","w")
                    DockergIgnoreFile.write(DockergIgnoreData)
                    DockergIgnoreFile.close()
                    self.UI.ProgressBar.setValue(70)
                    Dockerfile = open(os.path.join(ProjectDir,"Templates/DockerFiles/JavaDockerfile"),"r")
                    DockerfileData = Dockerfile.read()
                    Dockerfile.close()
                    Dockerfile = open("Dockerfile","w")
                    Dockerfile.write(DockerfileData + "\n" + f'RUN javac {self.NameScript}.java')
                    Dockerfile.close()
                self.UI.ProgressBar.setValue(100)
                self.Notification("Артём - голосовой ассистент","Java проект успешно создан.")
                logger.info("Java проект успешно создан.")
            else:
                self.UI.ErrorNameProject.setText("Папка с таким именем уже существует")
                self.UI.ErrorNameProject.setVisible(True)
        except Exception as e:
            logger.error(e)

    def CProject(self):
        print("C")
        try:
            if self.NameScript.endswith(".c") == True:
                self.NameScript = self.NameScript.replace(".c","")
            os.chdir(self.Directory)
            os.mkdir(self.NameProject)
            os.chdir(os.path.join(self.Directory,self.NameProject))
            if not os.path.exists(self.NameProject):
                self.UI.ProgressBar.setVisible(True)
                self.UI.ProgressBar.setValue(5)
                GitIgnoreFile = open(os.path.join(ProjectDir,"Templates/GitIgnore/C.gitignore"),"r")
                GitIgnoreData = GitIgnoreFile.read()
                GitIgnoreFile.close()
                GitIgnoreFile = open(".gitignore","w")
                GitIgnoreFile.write(GitIgnoreData)
                GitIgnoreFile.close()
                self.UI.ProgressBar.setValue(20)
                ScriptFile = open(os.path.join(ProjectDir,"Templates/CTemplate.c"),"r")
                ScriptFileData = ScriptFile.read()
                ScriptFile.close()
                ScriptFile = open(f"{self.NameScript}.c","w")
                ScriptFile.write(ScriptFileData)
                ScriptFile.close()
                os.system(f'gcc -o {self.NameScript} {self.NameScript}.c')
                self.UI.ProgressBar.setValue(50)
                if self.UI.DockerCheckBox.isChecked() == True:
                    DockergIgnoreFile = open(os.path.join(ProjectDir,"Templates/.dockerignore"),"r")
                    DockergIgnoreData = DockergIgnoreFile.read()
                    DockergIgnoreFile.close()
                    DockergIgnoreFile = open(".dockerignore","w")
                    DockergIgnoreFile.write(DockergIgnoreData)
                    DockergIgnoreFile.close()
                    self.UI.ProgressBar.setValue(70)
                    Dockerfile = open(os.path.join(ProjectDir,"Templates/DockerFiles/CDockerfile"),"r")
                    DockerfileData = Dockerfile.read()
                    Dockerfile.close()
                    Dockerfile = open("Dockerfile","w")
                    Dockerfile.write(DockerfileData + "\n" + f'RUN gcc -o {self.NameScript} {self.NameScript}.c' + "\n" + f'CMD ["./{self.NameScript}"]')
                    Dockerfile.close()
                self.UI.ProgressBar.setValue(100)
                self.Notification("Артём - голосовой ассистент","C проект успешно создан.")
                logger.info("C проект успешно создан.")
            else:
                self.UI.ErrorNameProject.setText("Папка с таким именем уже существует")
                self.UI.ErrorNameProject.setVisible(True)
        except Exception as e:
            logger.error(e)
    def GoProject(self):
        print("Go")
        try:
            if self.NameScript.endswith(".go") == True:
                self.NameScript = self.NameScript.replace(".go","")
            os.chdir(self.Directory)
            os.mkdir(self.NameProject)
            os.chdir(os.path.join(self.Directory,self.NameProject))
            if not os.path.exists(self.NameProject):
                self.UI.ProgressBar.setVisible(True)
                self.UI.ProgressBar.setValue(5)
                GitIgnoreFile = open(os.path.join(ProjectDir,"Templates/GitIgnore/Go.gitignore"),"r")
                GitIgnoreData = GitIgnoreFile.read()
                GitIgnoreFile.close()
                GitIgnoreFile = open(".gitignore","w")
                GitIgnoreFile.write(GitIgnoreData)
                GitIgnoreFile.close()
                self.UI.ProgressBar.setValue(20)
                ScriptFile = open(os.path.join(ProjectDir,"Templates/GoTemplate.go"),"r")
                ScriptFileData = ScriptFile.read()
                ScriptFile.close()
                ScriptFile = open(f"{self.NameScript}.go","w")
                ScriptFile.write(ScriptFileData)
                ScriptFile.close()
                os.system(f'go mod init {self.NameScript}')
                self.UI.ProgressBar.setValue(40)
                os.system(f"go build {self.NameScript}")
                self.UI.ProgressBar.setValue(50)
                if self.UI.DockerCheckBox.isChecked() == True:
                    DockergIgnoreFile = open(os.path.join(ProjectDir,"Templates/.dockerignore"),"r")
                    DockergIgnoreData = DockergIgnoreFile.read()
                    DockergIgnoreFile.close()
                    DockergIgnoreFile = open(".dockerignore","w")
                    DockergIgnoreFile.write(DockergIgnoreData)
                    DockergIgnoreFile.close()
                    self.UI.ProgressBar.setValue(70)
                    Dockerfile = open(os.path.join(ProjectDir,"Templates/DockerFiles/GoDockerfile"),"r")
                    DockerfileData = Dockerfile.read()
                    Dockerfile.close()
                    Dockerfile = open("Dockerfile","w")
                    Dockerfile.write(DockerfileData + "\n" + f'RUN go mod init {self.NameScript}'+ "\n" + f"RUN go build {self.NameScript}" + "\n" + f'CMD ["./{self.NameScript}"]')
                    Dockerfile.close()
                self.UI.ProgressBar.setValue(100)
                self.Notification("Артём - голосовой ассистент","Go проект успешно создан.")
                logger.info("Go проект успешно создан.")
            else:
                self.UI.ErrorNameProject.setText("Папка с таким именем уже существует")
                self.UI.ErrorNameProject.setVisible(True)
        except Exception as e:
            logger.error(e)

    def RustProject(self):
        print("Rust")
        try:
            if self.NameScript.endswith(".rs") == True:
                self.NameScript = self.NameScript.replace(".rs","")
            os.chdir(self.Directory)
            os.mkdir(self.NameProject)
            os.chdir(os.path.join(self.Directory,self.NameProject))
            if not os.path.exists(self.NameProject):
                self.UI.ProgressBar.setVisible(True)
                self.UI.ProgressBar.setValue(5)
                GitIgnoreFile = open(os.path.join(ProjectDir,"Templates/GitIgnore/Rust.gitignore"),"r")
                GitIgnoreData = GitIgnoreFile.read()
                GitIgnoreFile.close()
                GitIgnoreFile = open(".gitignore","w")
                GitIgnoreFile.write(GitIgnoreData)
                GitIgnoreFile.close()
                self.UI.ProgressBar.setValue(20)
                ScriptFile = open(os.path.join(ProjectDir,"Templates/RustTemplate.rs"),"r")
                ScriptFileData = ScriptFile.read()
                ScriptFile.close()
                ScriptFile = open(f"{self.NameScript}.rs","w")
                ScriptFile.write(ScriptFileData)
                ScriptFile.close()
                os.system(f"rustc {self.NameScript}.rs")
                self.UI.ProgressBar.setValue(50)
                if self.UI.DockerCheckBox.isChecked() == True:
                    DockergIgnoreFile = open(os.path.join(ProjectDir,"Templates/.dockerignore"),"r")
                    DockergIgnoreData = DockergIgnoreFile.read()
                    DockergIgnoreFile.close()
                    DockergIgnoreFile = open(".dockerignore","w")
                    DockergIgnoreFile.write(DockergIgnoreData)
                    DockergIgnoreFile.close()
                    self.UI.ProgressBar.setValue(70)
                    Dockerfile = open(os.path.join(ProjectDir,"Templates/DockerFiles/RustDockerfile"),"r")
                    DockerfileData = Dockerfile.read()
                    Dockerfile.close()
                    Dockerfile = open("Dockerfile","w")
                    Dockerfile.write(DockerfileData + "\n" + f"RUN rustc {self.NameScript}.rs" + "\n" + f'CMD ["./{self.NameScript}"]')
                    Dockerfile.close()
                self.UI.ProgressBar.setValue(100)
                self.Notification("Артём - голосовой ассистент","Rust проект успешно создан.")
                logger.info("Rust проект успешно создан.")
            else:
                self.UI.ErrorNameProject.setText("Папка с таким именем уже существует")
                self.UI.ErrorNameProject.setVisible(True)
        except Exception as e:
            logger.error(e)

    def DjangoProject(self):
        print("Django")
        try:
            os.chdir(self.Directory)
            if not os.path.exists(self.NameProject):
                self.UI.ProgressBar.setVisible(True)
                self.UI.ProgressBar.setValue(5)
                if platform.system() == "Windows":
                    os.system("pip install django django-restframework")
                elif platform.system() == "Linux":
                    os.system("pip3 install django django-restframework")
                elif platform.system() == "Darwin":
                    os.system("pip3 install django django-restframework")

                os.system(f"django-admin startproject {self.NameProject}")
                # self.UI.ProgressBar.setValue(15)
                os.chdir(os.path.join(self.Directory,self.NameProject))
                GitIgnoreFile = open(os.path.join(ProjectDir,"Templates/GitIgnore/Python.gitignore"),"r")
                GitIgnoreData = GitIgnoreFile.read()
                GitIgnoreFile.close()
                GitIgnoreFile = open(".gitignore","w")
                GitIgnoreFile.write(GitIgnoreData)
                GitIgnoreFile.close()
                self.UI.ProgressBar.setValue(20)
                if self.UI.EnvCheckBox.isChecked() == True:
                    if platform.system() == "Windows":
                        os.system("python -m venv env")
                        EnviromentCommand = f"{os.getcwd()}/env/Scripts/activate & pip install -U --upgrade pip setuptools wheel & pip install django django-restframework pillow django-debug-toolbar  static3 dj-static & pip freeze > requirements.txt & deactivate"
                        os.system(EnviromentCommand)
                    elif platform.system() == "Linux":
                        os.system("python3 -m venv env")
                        EnviromentCommand = f"{os.getcwd()}/env/bin/activate & pip install -U --upgrade pip setuptools wheel & pip install django django-restframework pillow django-debug-toolbar  static3 dj-static & pip freeze > requirements.txt & deactivate"
                        os.system(EnviromentCommand)
                    elif platform.system() == "Darwin":
                        os.system("pytho3n -m venv env")
                        EnviromentCommand = f"{os.getcwd()}/env/bin/activate & pip install -U --upgrade pip setuptools wheel & pip install django django-restframework pillow django-debug-toolbar  static3 dj-static & pip freeze > requirements.txt & deactivate"
                        os.system(EnviromentCommand)
                self.UI.ProgressBar.setValue(50)
                if self.UI.DockerCheckBox.isChecked() == True:
                    DockergIgnoreFile = open(os.path.join(ProjectDir,"Templates/.dockerignore"),"r")
                    DockergIgnoreData = DockergIgnoreFile.read()
                    DockergIgnoreFile.close()
                    DockergIgnoreFile = open(".dockerignore","w")
                    DockergIgnoreFile.write(DockergIgnoreData)
                    DockergIgnoreFile.close()
                    self.UI.ProgressBar.setValue(70)
                    DockerFile = open(os.path.join(ProjectDir,"Templates/DockerFiles/DjangoDockerfile"),"r")
                    DockerFileData = DockerFile.read()
                    DockerFile.close()
                    DockerFile = open("Dockerfile","w")
                    DockerFile.write(DockerFileData)
                    DockerFile.close()
                self.UI.ProgressBar.setValue(100)
                self.Notification("Артём - голосовой ассистент","Django проект успешно создан.")
            else:
                self.UI.ErrorNameProject.setText("Папка с таким именем уже существует")
                self.UI.ErrorNameProject.setVisible(True)
        except Exception as e:
            logger.error(e)

    def NodeJSProject(self):
        try:
            if self.NameScript.endswith(".js") == True:
                self.NameScript = self.NameScript.replace(".js","")
            print("NodeJS")
            os.chdir(self.Directory)
            os.mkdir(self.NameProject)
            os.chdir(os.path.join(self.Directory,self.NameProject))
            if not os.path.exists(self.NameProject):
                self.UI.ProgressBar.setVisible(True)
                self.UI.ProgressBar.setValue(5)
                GitIgnoreFile = open(os.path.join(ProjectDir,"Templates/GitIgnore/NodeJS.gitignore"),"r")
                GitIgnoreData = GitIgnoreFile.read()
                GitIgnoreFile.close()
                GitIgnoreFile = open(".gitignore","w")
                GitIgnoreFile.write(GitIgnoreData)
                GitIgnoreFile.close()
                self.UI.ProgressBar.setValue(20)
                PackageFile = open("package.json","w")
                json.dump(
                    {
                        "name": f"{self.NameProject}",
                        "version": "1.0.0",
                        "description": f"{self.NameProject}",
                        "author": "",
                        "main": f"{self.NameScript}.js",
                        "scripts": {
                            "start": f"node {self.NameScript}.js"
                        },
                        "dependencies": {
                            "express": "^4.16.1"
                        }
                    },PackageFile,sort_keys=True,ensure_ascii=False,indent=2
                )
                PackageFile.close()
                self.UI.ProgressBar.setValue(50)
                ScriptFile = open(os.path.join(ProjectDir,"Templates/NodeJSTemplate.js"),"r")
                ScriptFileData = ScriptFile.read()
                ScriptFile.close()
                ScriptFile = open(f"{self.NameScript}.js","w")
                ScriptFile.write(ScriptFileData)
                ScriptFile.close()
                os.system("npm install")
                self.UI.ProgressBar.setValue(60)
                if self.UI.DockerCheckBox.isChecked() == True:
                    DockergIgnoreFile = open(os.path.join(ProjectDir,"Templates/.dockerignore"),"r")
                    DockergIgnoreData = DockergIgnoreFile.read()
                    DockergIgnoreFile.close()
                    DockergIgnoreFile = open(".dockerignore","w")
                    DockergIgnoreFile.write(DockergIgnoreData)
                    DockergIgnoreFile.close()
                    self.UI.ProgressBar.setValue(70)
                    Dockerfile = open(os.path.join(ProjectDir,"Templates/DockerFiles/NodeJSDockerfile"),"r")
                    DockerfileData = Dockerfile.read()
                    Dockerfile.close()
                    Dockerfile = open("Dockerfile","w")
                    Dockerfile.write(DockerfileData + "\n" + f'CMD [ "node", "{self.NameScript}.js" ]')
                    Dockerfile.close()
                self.UI.ProgressBar.setValue(100)
                self.Notification("Артём - голосовой ассистент","NodeJS проект успешно создан.")
                logger.info("NodeJS проект успешно создан.")
            else:
                self.UI.ErrorNameProject.setText("Папка с таким именем уже существует")
                self.UI.ErrorNameProject.setVisible(True)
        except Exception as e:
            logger.error(e)

    def CppProject(self):
        try:
            if self.NameScript.endswith(".cpp") == True:
                self.NameScript = self.NameScript.replace(".cpp","")
            print("C++")
            os.chdir(self.Directory)
            os.mkdir(self.NameProject)
            os.chdir(os.path.join(self.Directory,self.NameProject))
            if not os.path.exists(self.NameProject):
                self.UI.ProgressBar.setVisible(True)
                self.UI.ProgressBar.setValue(5)
                GitIgnoreFile = open(os.path.join(ProjectDir,"Templates/GitIgnore/C++.gitignore"),"r")
                GitIgnoreData = GitIgnoreFile.read()
                GitIgnoreFile.close()
                GitIgnoreFile = open(".gitignore","w")
                GitIgnoreFile.write(GitIgnoreData)
                GitIgnoreFile.close()
                self.UI.ProgressBar.setValue(20)
                ScriptFile = open(os.path.join(ProjectDir,"Templates/CppTemplate.cpp"),"r")
                ScriptFileData = ScriptFile.read()
                ScriptFile.close()
                ScriptFile = open(f"{self.NameScript}.cpp","w")
                ScriptFile.write(ScriptFileData)
                ScriptFile.close()
                self.UI.ProgressBar.setValue(50)
                os.system(f'g++ -o {self.NameScript} {self.NameScript}.cpp')
                if self.UI.DockerCheckBox.isChecked() == True:
                    DockergIgnoreFile = open(os.path.join(ProjectDir,"Templates/.dockerignore"),"r")
                    DockergIgnoreData = DockergIgnoreFile.read()
                    DockergIgnoreFile.close()
                    DockergIgnoreFile = open(".dockerignore","w")
                    DockergIgnoreFile.write(DockergIgnoreData)
                    DockergIgnoreFile.close()
                    self.UI.ProgressBar.setValue(70)
                    Dockerfile = open(os.path.join(ProjectDir,"Templates/DockerFiles/CppDockerfile"),"r")
                    DockerfileData = Dockerfile.read()
                    Dockerfile.close()
                    Dockerfile = open("Dockerfile","w")
                    Dockerfile.write(DockerfileData + "\n" + f'RUN g++ -o {self.NameScript} {self.NameScript}.cpp' + "\n" + f'CMD ["./{self.NameScript}"]')
                    Dockerfile.close()
                self.UI.ProgressBar.setValue(100)
                self.Notification("Артём - голосовой ассистент","C++ проект успешно создан.")
                logger.info("C++ проект успешно создан.")
            else:
                self.UI.ErrorNameProject.setText("Папка с таким именем уже существует")
                self.UI.ErrorNameProject.setVisible(True)
        except Exception as e:
            logger.error(e)

    def PythonProject(self):
        try:
            if self.NameScript.endswith(".py") == True:
                self.NameScript = self.NameScript.replace(".py","")
            print("Python")
            os.chdir(self.Directory)
            if not os.path.exists(self.NameProject):
                self.UI.ProgressBar.setVisible(True)
                self.UI.ProgressBar.setValue(5)
                os.mkdir(self.NameProject)
                os.chdir(os.path.join(self.Directory,self.NameProject))
                GitIgnoreFile = open(os.path.join(ProjectDir,"Templates/GitIgnore/Python.gitignore"),"r")
                GitIgnoreData = GitIgnoreFile.read()
                GitIgnoreFile.close()
                GitIgnoreFile = open(".gitignore","w")
                GitIgnoreFile.write(GitIgnoreData)
                GitIgnoreFile.close()
                self.UI.ProgressBar.setValue(20)
                if self.UI.EnvCheckBox.isChecked() == True:
                    if platform.system() == "Windows":
                        os.system("python -m venv env")
                        EnviromentCommand = f"{os.getcwd()}/env/Scripts/activate & pip install -U --upgrade pip setuptools wheel loguru pillow numpy matplotlib cx-freeze & pip freeze > requirements.txt & deactivate"
                        os.system(EnviromentCommand)
                    elif platform.system() == "Linux":
                        os.system("python3 -m venv env")
                        EnviromentCommand = f"{os.getcwd()}/env/bin/activate & pip install -U --upgrade pip setuptools wheel loguru pillow numpy matplotlib cx-freeze & pip freeze > requirements.txt & deactivate"
                        os.system(EnviromentCommand)
                    elif platform.system() == "Darwin":
                        os.system("python3 -m venv env")
                        EnviromentCommand = f"{os.getcwd()}/env/bin/activate & pip install -U --upgrade pip setuptools wheel loguru pillow numpy matplotlib cx-freeze & pip freeze > requirements.txt & deactivate"
                        os.system(EnviromentCommand)
                self.UI.ProgressBar.setValue(50)
                if self.UI.DockerCheckBox.isChecked() == True:
                    DockergIgnoreFile = open(os.path.join(ProjectDir,"Templates/.dockerignore"),"r")
                    DockergIgnoreData = DockergIgnoreFile.read()
                    DockergIgnoreFile.close()
                    DockergIgnoreFile = open(".dockerignore","w")
                    DockergIgnoreFile.write(DockergIgnoreData)
                    DockergIgnoreFile.close()
                    self.UI.ProgressBar.setValue(70)
                    Dockerfile = open(os.path.join(ProjectDir,"Templates/DockerFiles/PythonDockerfile"),"r")
                    DockerfileData = Dockerfile.read()
                    Dockerfile.close()
                    Dockerfile = open("Dockerfile","w")
                    Dockerfile.write(DockerfileData + "\n" + f'CMD [ "python3", "{self.NameScript}.py"]')
                    Dockerfile.close()
                self.UI.ProgressBar.setValue(100)
                self.Notification("Артём - голосовой ассистент","Python проект успешно создан.")
                logger.info("Python проект успешно создан.")
            else:
                self.UI.ErrorNameProject.setText("Папка с таким именем уже существует")
                self.UI.ErrorNameProject.setVisible(True)
        except Exception as e:
            logger.error(e)

    def Create(self):
        self.NameProject = self.UI.NameProjectEdit.text()
        self.NameScript = self.UI.NameMainScript.text()
        if len(self.NameProject.split()) == 0:
            self.UI.ErrorNameProject.setHidden(False)
        else:
            self.UI.ErrorNameProject.setHidden(True)
        if len(self.NameScript.split()) == 0 and self.TypeProject != "django":
            self.UI.ErrorMainScript.setHidden(False)
        else:
            self.UI.ErrorMainScript.setHidden(True)
        # Проверка типа проекта в словаре 
        if self.TypeProject in self.TypeFunctions:
            CreateThread = Thread(target = self.TypeFunctions[self.TypeProject])
            CreateThread.start()
        else:
            sys.exit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    Widgets = CreateProject(type="django_project")
    sys.exit(app.exec())