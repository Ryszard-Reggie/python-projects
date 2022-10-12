from tkinter import *

error_message = "ERROR!"


def insert_sing(sign):
    if entry_field.get() != error_message:
        last_string = entry_field.get()
        entry_field.delete(0, END)
        entry_field.insert(0, str(last_string) + str(sign))
    else:
        entry_field.delete(0, END)
        entry_field.insert(0, str(sign))


def enter_result():
    try:
        math_operation = entry_field.get()
        entry_field.delete(0, END)
        entry_field.insert(0, eval(math_operation))
    except:
        entry_field.insert(0, error_message)


def clear_sign():
    entry_field.delete(len(entry_field.get())-1, END)


def clear_all():
    entry_field.delete(0, END)


root = Tk()
root.title("Prosty kalkulator")

# Pole wprowadzania: ===================================================================================================

entry_field = Entry(root, borderwidth=3, width=40)
entry_field.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

# Operacje matematyczne: ===============================================================================================

addition = Button(text="+", padx=27, pady=21, borderwidth=3, command=lambda: insert_sing('+'))
subtraction = Button(text="-", padx=27, pady=21, borderwidth=3, command=lambda: insert_sing('-'))
multiplication = Button(text="*", padx=27, borderwidth=3, pady=21, command=lambda: insert_sing('*'))
division = Button(text="/", padx=27, pady=21, borderwidth=3, command=lambda: insert_sing('/'))
dot = Button(text=".", padx=27, pady=21, borderwidth=3, command=lambda: insert_sing('.'))

addition.grid(row=1, column=0)
subtraction.grid(row=1, column=1)
multiplication.grid(row=1, column=2)
division.grid(row=1, column=3)
dot.grid(row=5, column=2)

# Cyfry: ===============================================================================================================

number0 = Button(text="0", padx=54, pady=21, borderwidth=3, command=lambda: insert_sing('0'))
number1 = Button(text="1", padx=27, pady=21, borderwidth=3, command=lambda: insert_sing('1'))
number2 = Button(text="2", padx=27, pady=21, borderwidth=3, command=lambda: insert_sing('2'))
number3 = Button(text="3", padx=27, pady=21, borderwidth=3, command=lambda: insert_sing('3'))
number4 = Button(text="4", padx=27, pady=21, borderwidth=3, command=lambda: insert_sing('4'))
number5 = Button(text="5", padx=27, pady=21, borderwidth=3, command=lambda: insert_sing('5'))
number6 = Button(text="6", padx=27, pady=21, borderwidth=3, command=lambda: insert_sing('6'))
number7 = Button(text="7", padx=27, pady=21, borderwidth=3, command=lambda: insert_sing('7'))
number8 = Button(text="8", padx=27, pady=21, borderwidth=3, command=lambda: insert_sing('8'))
number9 = Button(text="9", padx=27, pady=21, borderwidth=3, command=lambda: insert_sing('9'))

number0.grid(row=5, column=0, columnspan=2)
number1.grid(row=4, column=0)
number2.grid(row=4, column=1)
number3.grid(row=4, column=2)
number4.grid(row=3, column=0)
number5.grid(row=3, column=1)
number6.grid(row=3, column=2)
number7.grid(row=2, column=0)
number8.grid(row=2, column=1)
number9.grid(row=2, column=2)

# Wynik i czyszczenie: =================================================================================================

clear_sign = Button(text='<x', padx=27, pady=21, borderwidth=3, command=clear_sign)
clear_all = Button(text='<<<x', padx=27, pady=21, borderwidth=3, command=clear_all)
result = Button(text='=', padx=27, pady=42, borderwidth=3, command=enter_result)

clear_sign.grid(row=2, column=3)
clear_all.grid(row=3, column=3)
result.grid(row=4, rowspan=2, column=3)

root.mainloop()
