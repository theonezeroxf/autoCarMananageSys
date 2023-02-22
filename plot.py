import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import DB.ConSql as dataset
def tryPic():
    rows: tuple = dataset.selectAllPlace()
    plt.figure(figsize=(6, 4), dpi=100)
    # cellText = None, cellColours = None, cellLoc = 'right',
    # colWidths = None, rowLabels = None, rowColours = None,
    # rowLoc = 'left', colLabels = None, colColours = None,
    # colLoc = 'center', loc = 'bottom', bbox = None, edges = 'closed',
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.table(loc='center', cellText=rows, colWidths=[0.25] * 4, colLabels=['车位号', '车牌号', '停车次数', '停车时长'])
    plt.show()

def gui2_try():
    root = tk.Tk()
    w=tk.Toplevel(root)
    w.title('表格')
    w.geometry('600x400')
    fig1=plt.figure(figsize=(6, 4), dpi=100)
    ax=fig1.add_subplot(111)

    canvas=FigureCanvasTkAgg(fig1,w)
    canvas.draw()
    canvas.get_tk_widget().grid()
    plt.rcParams['font.sans-serif'] = ['SimHei']

    global rows
    rows= dataset.selectAllPlace()


    ax.table(loc='center', cellText=rows, colWidths=[0.25] * 4, colLabels=['车位号', '车牌号', '停车次数', '停车时长'])
    ax.axis('off')
    root.mainloop()
if __name__=="__main__":
    gui2_try()

