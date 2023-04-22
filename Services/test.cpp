#include <SDL/include/SDL.h>  
#include <Windows.h>
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);  
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR szCmdLine, int iCmdShow)  
{  
    TCHAR appname[] = TEXT("Move Window");  
    MSG msg;  
    HWND hwnd;  
    WNDCLASS wndclass;  
    wndclass.cbClsExtra = 0;  
    wndclass.cbWndExtra = 0;  
    wndclass.hbrBackground = (HBRUSH) GetStockObject(WHITE_BRUSH);  
    wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);  
    wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);  
    wndclass.hInstance = hInstance;  
    wndclass.lpfnWndProc = WndProc;  
    wndclass.lpszClassName = appname;  
    wndclass.lpszMenuName = NULL;  
    wndclass.style = CS_HREDRAW | CS_VREDRAW;  
    if (!RegisterClass( & wndclass))  
    {  
        MessageBox(NULL, TEXT("Class not registered"), TEXT("Error...."), MB_OK);  
    }  
    hwnd = CreateWindow(appname,  
        appname,  
        WS_OVERLAPPEDWINDOW,  
        300,  
        200,  
        400,  
        300,  
        NULL,  
        NULL,  
        hInstance,  
        NULL);  
    ShowWindow(hwnd, iCmdShow);  
    UpdateWindow(hwnd);  
    while (GetMessage( & msg, NULL, 0, 0))  
    {  
        TranslateMessage( & msg);  
        DispatchMessage( & msg);  
    }  
    return msg.wParam;  
}  
TCHAR * getString(int type)  
{  
    TCHAR * str = L "";  
    switch (type)  
    {  
        case 1:  
            str = L "UP     :      Move Window Upward";  
            break;  
        case 2:  
            str = L "Down     :      Move Window Downward";  
            break;  
        case 3:  
            str = L "Left     :      Move Window to Left";  
            break;  
        case 4:  
            str = L "Right     :      Move Window to Right";  
            break;  
        case 5:  
            str = L " +     :      Increase Size of Window";  
            break;  
        case 6:  
            str = L " -     :      Decrease Size of Window";  
            break;  
    }  
    return str;  
}  
LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)  
{  
    HDC hdc;  
    RECT rc;  
    static int X, Y;  
    PAINTSTRUCT ps;  
    HWND hwndSmaller;  
    switch (message)  
    {  
        case WM_CREATE:  
            return 0;  
            break;  
        case WM_PAINT:  
            hdc = BeginPaint(hwnd, & ps);  
            SetTextColor(hdc, RGB(0, 0, 180));  
            TextOut(hdc, 5, 5, getString(1), 32);  
            TextOut(hdc, 5, 25, getString(2), 36);  
            TextOut(hdc, 5, 45, getString(3), 35);  
            TextOut(hdc, 5, 65, getString(4), 37);  
            TextOut(hdc, 5, 85, getString(5), 37);  
            TextOut(hdc, 5, 105, getString(6), 37);  
            EndPaint(hwnd, & ps);  
            return 0;  
            break;  
        case WM_SIZE:  
            X = LOWORD(lParam);  
            Y = HIWORD(lParam);  
            return 0;  
            break;  
            // key down    
        case WM_KEYDOWN:  
            switch (wParam)  
            {  
                case VK_LEFT:  
                    GetWindowRect(hwnd, & rc);  
                    rc.left -= X / 20;  
                    rc.right -= X / 20;  
                    MoveWindow(hwnd, rc.left, rc.top, rc.right - rc.left, rc.bottom - rc.top, TRUE);  
                    break;  
                case VK_RIGHT:  
                    GetWindowRect(hwnd, & rc);  
                    rc.left += X / 20;  
                    rc.right += X / 20;  
                    MoveWindow(hwnd, rc.left, rc.top, rc.right - rc.left, rc.bottom - rc.top, TRUE);  
                    break;  
                case VK_UP:  
                    GetWindowRect(hwnd, & rc);  
                    rc.bottom -= X / 20;  
                    rc.top -= X / 20;  
                    MoveWindow(hwnd, rc.left, rc.top, rc.right - rc.left, rc.bottom - rc.top, TRUE);  
                    break;  
                case VK_DOWN:  
                    GetWindowRect(hwnd, & rc);  
                    rc.bottom += X / 20;  
                    rc.top += X / 20;  
                    MoveWindow(hwnd, rc.left, rc.top, rc.right - rc.left, rc.bottom - rc.top, TRUE);  
                    break;  
                case VK_ESCAPE:  
                    PostQuitMessage(EXIT_SUCCESS);  
                    return 0;  
            }  
            return 0;  
            break;  
        case WM_CHAR:  
            switch (wParam)  
            {  
                //increase size of window    
                case '+':  
                    GetWindowRect(hwnd, & rc);  
                    rc.left -= X / 20;  
                    rc.right += X / 20;  
                    rc.top -= Y / 20;  
                    rc.bottom += Y / 20;  
                    MoveWindow(hwnd, rc.left, rc.top, rc.right - rc.left, rc.bottom - rc.top, TRUE);  
                    break;  
                    // decrease size of window    
                case '-':  
                    GetWindowRect(hwnd, & rc);  
                    rc.left += X / 20;  
                    rc.right -= X / 20;  
                    rc.top += Y / 20;  
                    rc.bottom -= Y / 20;  
                    MoveWindow(hwnd, rc.left, rc.top, rc.right - rc.left, rc.bottom - rc.top, TRUE);  
                    break;  
            }  
            return 0;  
            break;  
        case WM_DESTROY:  
            PostQuitMessage(EXIT_SUCCESS);  
            return 0;  
    }  
    return DefWindowProc(hwnd, message, wParam, lParam);  
} 