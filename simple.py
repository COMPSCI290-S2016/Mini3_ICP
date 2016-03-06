#A simple file for testing wxPython
import wx

if __name__ == '__main__':
    app = wx.App()

    frame = wx.Frame(None, -1, 'simple.py')
    frame.Show()

    app.MainLoop()
