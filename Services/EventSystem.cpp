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

using namespace std;
using namespace std::chrono;

void print(string sentence) {
    cout << sentence << endl;
}
map <string, string> Keys = {{"1", "Ctrl+A"}, {"2", "Ctrl+B"}, {"3", "Ctrl+C"},{"4", "Ctrl+D"},{"5", "Ctrl+E"},{"6", "Ctrl+F"},{"7", "Ctrl+G"},{"8", "Ctrl+H"},{"9", "Ctrl+I"},{"10", "Ctrl+J"},{"11", "Ctrl+K"},{"12", "Ctrl+L"},{"13", "Ctrl+M"},{"14", "Ctrl+N"},{"15", "Ctrl+O"},{"16", "Ctrl+P"},{"17", "Ctrl+Q"},{"18", "Ctrl+R"},{"19", "Ctrl+S"},{"20", "Ctrl+T"},{"21", "Ctrl+U"},{"22", "Ctrl+V"},{"23", "Ctrl+W"},{"24", "Ctrl+X"},{"25", "Ctrl+Y"},{"26", "Ctrl+Z"},{"48", "0"},{"49", "1"},{"50", "2"},{"51", "3"},{"52", "4"},{"53", "5"},{"54", "6"},{"55", "7"},{"56", "8"},{"57", "9"},{"65", "A"},{"66", "B"},{"67", "C"},{"68", "D"},{"69", "E"},{"70", "F"},{"71", "G"},{"72", "H"},{"73", "I"},{"74", "J"},{"75", "K"},{"76", "L"},{"77", "M"},{"78", "N"},{"79", "O"},{"80", "P"},{"81", "Q"},{"82", "R"},{"83", "S"},{"84", "T"},{"85", "U"},{"86", "V"},{"87", "W"},{"88", "X"},{"89", "Y"},{"90", "Z"},{"5", "Ctrl+E"},};

class EventSystem {
    public:
        void init() {

        }

        EventSystem() {
            print("EventSystem has been initialized.");
        }
        ~EventSystem() {
            print("EventSystem has been killed.");
        }
    private:
        void ReadConfig() {

        }
};
int main() {
    setlocale(LC_ALL, "RU");
    char name_key;
    int ascii_value;
    while(true)
    {
        name_key = getch();
        ascii_value = name_key;
        if (Keys.count(to_string(ascii_value))) {
            printf("Key:%d",ascii_value);
        }
    }
    EventSystem event_system;
    event_system.init(); 
    system("pause");
    return 0;
}