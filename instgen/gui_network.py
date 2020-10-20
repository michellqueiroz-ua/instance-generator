import tkinter as tk
from tkinter import *

class window_network:
	def __init__(self, win):
		self.win = win
		self.lbl=Label(win, text="INFORM THE LOCATION'S NAME", fg='black', font=("Times", 16))
		self.lbl.place(x=80, y=50)
		self.txtfld=Entry(win, text="Entry the location name", bd=5)
		self.txtfld.place(x=80, y=100)
		self.btn=Button(win, text='OK', command = lambda: self.btnOKclick() or self.quit())
		self.btn.place(x=80, y=150)
				
	def btnOKclick(self):
		print(self.txtfld.get())

	def quit(self):
		self.win.destroy()

def gui_network():

	window=Tk()
	winnet=window_network(window)
	window.title('Retrieve Network')
	window.geometry("400x300+10+10")
	window.mainloop()