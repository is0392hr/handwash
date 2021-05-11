import tkinter
from tkinter import messagebox

# click時のイベント
def btn_click():
  messagebox.showinfo("メッセージ", "ボタンがクリックされました")

# 画面作成
tki = tkinter.Tk()
tki.geometry('300x200') # 画面サイズの設定
tki.title('ボタンのサンプル') # 画面タイトルの設定

# ボタンの作成
btn = tkinter.Button(tki, text='ボタン', command = btn_click)
btn.place(x=130, y=80) #ボタンを配置する位置の設定

# 画面をそのまま表示
tki.mainloop()