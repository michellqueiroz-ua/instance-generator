import tkinter as tk
from tkinter import simpledialog


def gui_network():
	
	ROOT = tk.Tk()

	ROOT.withdraw()
	# the input dialog
	USER_INP = simpledialog.askstring(title="Test",
	                                  prompt="What's your Name?:")

	# check it out
	print("Hello", USER_INP)