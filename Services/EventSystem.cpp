// Импорт стандартных библиотек
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <time.h> 
#include <exception>
#include <filesystem>
#include <conio.h>
#include <stdlib.h>
#include <ctime> 
#include <map>
#include <json/json.h>  
#include <string>
#include <cctype>
#include <algorithm>

// Проверка названия операционной системы и импортрование нужнух библиотек для этой системы
#if defined(__linux__)
    std::cout << "Linux" << '\n';
#elif __FreeBSD__
    std::cout << "FreeBSD" << '\n';
#elif __APPLE__
    std::cout << "macOS" << '\n';
#elif _WIN32
    #include <Windows.h>
#endif

using namespace std;
using namespace std::chrono;

// Глобльные переменные
string OS_NAME;
string value;
size_t index_key;
int LenghtArray;
string HotKeys[128] = {"ctrl + b","shift + a","alt + d","Shift + y"};
// map <string, string> Keys = {{"1", "Ctrl+A"}, {"2", "Ctrl+B"}, {"3", "Ctrl+C"},{"4", "Ctrl+D"},{"5", "Ctrl+E"},{"6", "Ctrl+F"},{"7", "Ctrl+G"},{"8", "Ctrl+H"},{"9", "Ctrl+I"},{"10", "Ctrl+J"},{"11", "Ctrl+K"},{"12", "Ctrl+L"},{"13", "Ctrl+M"},{"14", "Ctrl+N"},{"15", "Ctrl+O"},{"16", "Ctrl+P"},{"17", "Ctrl+Q"},{"18", "Ctrl+R"},{"19", "Ctrl+S"},{"20", "Ctrl+T"},{"21", "Ctrl+U"},{"22", "Ctrl+V"},{"23", "Ctrl+W"},{"24", "Ctrl+X"},{"25", "Ctrl+Y"},{"26", "Ctrl+Z"},{"48", "0"},{"49", "1"},{"50", "2"},{"51", "3"},{"52", "4"},{"53", "5"},{"54", "6"},{"55", "7"},{"56", "8"},{"57", "9"},{"65", "A"},{"66", "B"},{"67", "C"},{"68", "D"},{"69", "E"},{"70", "F"},{"71", "G"},{"72", "H"},{"73", "I"},{"74", "J"},{"75", "K"},{"76", "L"},{"77", "M"},{"78", "N"},{"79", "O"},{"80", "P"},{"81", "Q"},{"82", "R"},{"83", "S"},{"84", "T"},{"85", "U"},{"86", "V"},{"87", "W"},{"88", "X"},{"89", "Y"},{"90", "Z"},{"5", "Ctrl+E"},};
// Словарь со значениями ключей и их названиями
map <string,int> Keys = {{"ctrl+a",0x41},{"ctrl+b",0x42}};
// Функции
void print(string sentence) {
    cout << sentence << endl;
}

class EventSystem {
    public:
        void init() {

        }
        void CheckPress() {
            if (OS_NAME == "Windows") {
                LenghtArray = *(&HotKeys + 1) - HotKeys;
                for (int i=0;i < LenghtArray;i++) {
                    value = HotKeys[i];
                    cout << value << endl;
                }
            }
            else if (OS_NAME == "Linux") {

            }
            else if (OS_NAME == "macOS") {

            }
        }
        EventSystem() {
            printf("EventSystem has been initialized.");
        }
        ~EventSystem() {
            printf("EventSystem has been killed.");
        }
    private:
        void SetHotkey(string Key) {
            bool Ctrl = false;
            bool Alt = false;
            bool Shift = false;
            size_t position;
            Key = tolower(Key);
            index_key = Key.find("ctrl");
            if (index_key != string::npos) {
                Ctrl = true;
                Key = Key.replace(index_key,sizeof("ctrl"),"");
                while ((position = Key.find("+")) != std::string::npos) {
                    Key.replace(position,sizeof("+"),"");
                }
                Key.erase(std::remove_if(Key.begin(), Key.end(), ::isspace), Key.end());
                cout << index_key << endl;
                cout << "Ctrl is finded" << endl;
            }
            // if (Ctrl == false){
            if (true) {
                index_key = Key.find("alt");
                if (index_key != string::npos) {
                    Alt = true;
                    Key = Key.replace(index_key,sizeof("alt"),"");
                    while ((position = Key.find("+")) != std::string::npos) {
                        Key.replace(position,sizeof("+"),"");
                    }
                    Key.erase(std::remove_if(Key.begin(), Key.end(), ::isspace), Key.end());
                    cout << index_key << endl;
                    cout << "Alt is finded" << endl;
                }
            }
            // if (Ctrl == false and Alt == false) {
            if (true) {
                index_key = Key.find("Shift");
                if (index_key != string::npos) {
                    Shift = true;
                    Key = Key.replace(index_key,sizeof("shift"),"");
                    while ((position = Key.find("+")) != std::string::npos) {
                        Key.replace(position,sizeof("+"),"");
                    }
                    Key.erase(std::remove_if(Key.begin(), Key.end(), ::isspace), Key.end());
                    cout << index_key << endl;
                    cout << "Shift is finded" << endl;
                }
            }
            for (auto it = Keys.begin(); it != Keys.end(); it++) {
                std::cout << "{" << it->first << ", " << it->second << "}" << std::endl;
            }
        }
        void ReadConfig() {

        }
};
int main() {
    setlocale(LC_ALL, "RU");
    char name_key;
    int ascii_value;
    char ch;
	int flag = 1;
    // while(true)
    // {
    //     name_key = _getch();
    //     ascii_value = name_key;
    //     if (Keys.count(to_string(ascii_value))) {
    //         printf("Key:%d",ascii_value);
    //     }
    // }
    EventSystem event_system;
    event_system.init();
    #if defined(__linux__)
        OS_NAME = "Linux";
    #elif __FreeBSD__
        OS_NAME = "FreeBSD";
    #elif __APPLE__
        OS_NAME = "macOS";
    #elif _WIN32
        OS_NAME = "Windows";
    #endif
    event_system.CheckPress(); 
    cout << OS_NAME << endl;
    // while(true) {
        // if(GetKeyState(0x25) & 0x8000) {
		// 	std::cout << "Right Shift key pressed" << std::endl;
		// 	flag = 0;
        //     Sleep(1);
		// }
        
        
    // }
    RegisterHotKey(NULL,1,MOD_ALT,0x42);
    RegisterHotKey(NULL,1,MOD_CONTROL,0x44);
    
    MSG msg = {0};
    while (GetMessage(&msg, NULL, 0, 0) != 0)
    {
        if (msg.message == WM_HOTKEY && GetKeyState(0x44) & 0x8000)
        {
            printf("Ctrl+D");
            printf("WM_HOTKEY received\n");            
        }
        else if(msg.message == WM_HOTKEY && GetKeyState(0x42) & 0x8000) {
            printf("Alt+B");

        }
    }
    system("pause");
    return 0;
}