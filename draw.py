from graphviz import Digraph
from common_header import *
import traceback, sys
from PyQt5 import uic, QtGui
from PyQt5.QtWidgets import *

# for multithreading in pyQT
from PyQt5.QtCore import *


form_class = uic.loadUiType("draw.ui")[0]
# for multithreading in pyQT
class threadSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        No data
    error
        `tuple` (exctype, value, traceback.format_exc() )
    result
        `object` data returned from processing, anything
    progress
        `int` indicating % progress
    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

class multithread(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(multithread, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = threadSignals()

        # Add the callback to our kwargs
        #self.kwargs['progress_callback'] = self.signals.progress
        
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            #traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            #self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.k=0
        self.pfa_nbS = 3
        self.pfa_nbL = 3
        self.num_of_string = 1
        self.input_dpfa = None
        self.hamming = None
        self.sub_pfa = None
        self.dpfa = None
        self.current_tab = 3
        self.k_mps_list = []
        self.fname = None
        self.pfa_mode = 'random' # random or file
        self.lineEdit.returnPressed.connect(self.enter_pressed)
        self.lineEdit_2.returnPressed.connect(self.enter_pressed)
        self.lineEdit_5.returnPressed.connect(self.filling_table)
        self.pushButton.clicked.connect(self.pushButtonClicked)
        self.lineEdit_2.setText(str(self.k))
        self.lineEdit_3.setText(str(self.pfa_nbL))
        self.lineEdit_4.setText(str(self.pfa_nbS))
        self.lineEdit_5.setText(str(self.num_of_string))
        self.lineEdit_2.textChanged.connect(self.hamming_distance)
        self.lineEdit_3.textChanged.connect(self.nbL_setting)
        self.lineEdit_4.textChanged.connect(self.nbS_setting)
        self.lineEdit_5.textChanged.connect(self.num_of_string_setting)
        self.scrollArea_1.setStyleSheet("background-color:#ffffff;")
        self.scrollArea_2.setStyleSheet("background-color:#ffffff;")
        self.scrollArea_3.setStyleSheet("background-color:#ffffff;")
        self.scrollArea_4.setStyleSheet("background-color:#ffffff;")
        self.tableWidget.itemSelectionChanged.connect(self.clicked)
        self.tableWidget.setColumnWidth(1,900)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.radioButton_1.clicked.connect(self.radioButtonClicked)
        self.radioButton_2.clicked.connect(self.radioButtonClicked)
        #self.tabWidget.currentChanged.connect(self.tab_change)
        #QScroller.grabGesture(self.scrollArea_1.viewport(), QScroller.LeftMouseButtonGesture)
        #QScroller.grabGesture(self.scrollArea_2.viewport(), QScroller.LeftMouseButtonGesture)
        #QScroller.grabGesture(self.scrollArea_3.viewport(), QScroller.LeftMouseButtonGesture)
        QScroller.grabGesture(self.scrollArea_4.viewport(), QScroller.LeftMouseButtonGesture)
        #self.scrollArea_1.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        #self.scrollArea_2.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        #self.scrollArea_3.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        #self.scrollArea_4.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)

        # for multithreading in pyQT
        self.threadpool = QThreadPool()

    def radioButtonClicked(self):
        if self.radioButton_1.isChecked():
            self.pfa_mode = 'random'
            self.nbS_setting()
            self.nbL_setting()
        elif self.radioButton_2.isChecked():
            self.pfa_mode = 'file'

    def pushButtonClicked(self):
            self.fname = QFileDialog.getOpenFileName(self)
            print(self.fname)
            self.textBrowser_2.setText(self.fname[0])

    def nbL_setting(self):
        nbL = self.lineEdit_3.text()
        if nbL == '':
           nbL = '0' 
        self.pfa_nbL = int(nbL)

    def nbS_setting(self):
        nbS = self.lineEdit_4.text()
        if nbS == '':
           nbS = '0' 
        self.pfa_nbS = int(nbS)

    def num_of_string_setting(self):
        num_of_string = self.lineEdit_5.text()
        if num_of_string == '':
           num_of_string = '0' 
        self.num_of_string = int(num_of_string)

    def enter_pressed(self):
        start_time = time.time()
        # make pfa
        if self.pfa_mode == 'random':
            self.input_dpfa = PFA_utils.DPFA_generator(nbS = self.pfa_nbS, nbL = self.pfa_nbL)
            #random_dpfa_bu = self.input_dpfa
            print('random generation done')
        elif self.pfa_mode == 'file':
            if self.fname == None:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("NO FILE")
                msg.exec_()
                return
            self.input_dpfa = PFA_utils.parser(self.fname[0])
            self.pfa_nbS = self.input_dpfa.nbS
            self.pfa_nbL = self.input_dpfa.nbL
            print('read input file done')
        # make hamming automaton
        input_string = str(self.lineEdit.text())
        input_nbL = len(set(input_string))
        if input_nbL != self.pfa_nbL:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("error: nbL of PFA and nbL of DFA are different")
            msg.exec_()
            print('error: nbL of PFA and nbL of DFA are different')
            return
        alphabet = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'
        sigma = alphabet.split(' ')[0:self.pfa_nbL]
        self.hamming = PFA_utils.DFA_constructor(input_string, self.k, sigma)
        self.hamming.print()
        print('hamming done')    
        # intersect hamming & pfa -> sub_pfa
        start_intersect_time = time.time()
        ra = self.input_dpfa.intersect_with_DFA(self.hamming)
        end_intersect_time = time.time()
        self.sub_dpfa = PFA.PFA(ra.nbL, ra.nbS, ra.initial, ra.final, ra.transitions)
        print('intersecting dpfa done')
        """
        start_dp_time = time.time()
        random_dpfa_bu.k_MPS(input_string, self.k)
        end_dp_time = time.time()
        print('=========================================')
        print('intersect time:', end_intersect_time - start_intersect_time)
        print('dp time       :', end_dp_time - start_dp_time)
        print('=========================================')
        """
        # intersected Automata 나중에 지우자
        #self.drawing_thread(self.input_dpfa, 'input_dpfa')
        #self.drawing_thread(self.hamming, 'hamming')
        #self.drawing_thread(self.sub_dpfa, 'sub_dpfa')
        
        # normalize sub_pfa -> dpfa
        self.dpfa = PFA_utils.normalizer(self.sub_dpfa)
        print('normalizing done')
        ##self.dpfa = input_dpfa
        
        # initialize the table
        self.filling_table()
        end_time = time.time()
        load_time = end_time - start_time
        # update drawing
        # random PFA
        self.drawing(self.input_dpfa, 'input_dpfa')
        
        # hamming Automata
        self.drawing(self.hamming, 'hamming')
        
        # intersected Automata
        self.drawing(self.sub_dpfa, 'sub_dpfa')
        
        # normalized Automata
        self.drawing(self.dpfa, 'dpfa')

        self.textBrowser.setText(str(load_time)+' seconds')
    
    def tab_change(self,i): #changed!
        #print('before, after:', self.current_tab, i)
        if self.current_tab == 0:
            sa = self.scrollArea_1            
        if self.current_tab == 1:
            sa = self.scrollArea_2            
        if self.current_tab == 2:
            sa = self.scrollArea_3            
        if self.current_tab == 3:
            sa = self.scrollArea_4
        QScroller.ungrabGesture(sa.viewport())

        if i == 0:
            self.current_tab = 0
            sa = self.scrollArea_1
        elif i == 1:
            self.current_tab = 1
            sa = self.scrollArea_2
        elif i == 2:
            self.current_tab = 2
            sa = self.scrollArea_3
        elif i == 3:
            self.current_tab = 3
            sa = self.scrollArea_4
        QScroller.grabGesture(sa.viewport(), QScroller.LeftMouseButtonGesture)        

    def filling_table(self):
        if self.dpfa == None:
            self.enter_pressed()
            return
        # Do exact MPS on dpfa
        if self.dpfa.nbS != 0:
            self.k_mps_list = self.dpfa.n_MPS( self.num_of_string )

        self.tableWidget.clearContents()
        self.tableWidget.setRowCount(0)
        
        if self.dpfa.nbS == 0:
            self.tableWidget.insertRow( self.tableWidget.rowCount() )
            self.tableWidget.setItem( self.tableWidget.rowCount()-1, 1, QTableWidgetItem('NO MPS'))
        else:
            for i in range(min(self.num_of_string,len(self.k_mps_list))):
                self.tableWidget.insertRow( self.tableWidget.rowCount() )
                self.tableWidget.setItem( self.tableWidget.rowCount()-1, 0, QTableWidgetItem(str(round(self.dpfa.parse(self.k_mps_list[i]),10))))
                self.tableWidget.setItem( self.tableWidget.rowCount()-1, 1, QTableWidgetItem(self.k_mps_list[i]))


    def drawing(self, automata, file_name):
        thread = multithread(self.drawing_thread, automata, file_name)
        self.threadpool.start(thread)
        
    def drawing_thread(self, automata, file_name):
        if file_name == 'input_dpfa':
            makePNG(self.input_dpfa, 'input_dpfa')
            input_dpfa = QtGui.QPixmap("input_dpfa.png") 
            self.draw_1.setPixmap(input_dpfa)
            self.draw_1.resize(input_dpfa.size())

        elif file_name == 'hamming':
            makePNG(self.hamming, 'hamming')
            hamming = QtGui.QPixmap("hamming.png") 
            self.draw_2.setPixmap(hamming)
            self.draw_2.resize(hamming.size())

        elif file_name == 'sub_dpfa':
            makePNG(self.sub_dpfa, 'sub_dpfa')
            sub_dpfa = QtGui.QPixmap("sub_dpfa.png") 
            self.draw_3.setPixmap(sub_dpfa)
            self.draw_3.resize(sub_dpfa.size())
        
        elif file_name == 'dpfa':
            makePNG(self.dpfa, 'dpfa')
            dpfa = QtGui.QPixmap("dpfa.png") 
            self.draw_4.setPixmap(dpfa)
            self.draw_4.resize(dpfa.size())

    def hamming_distance(self):
        d = self.lineEdit_2.text()
        if d == '':
           d = '0' 
        self.k = int(d)

    def clicked(self):
        thread = multithread(self.clicked_thread)
        self.threadpool.start(thread)
    
    def clicked_thread(self):
        if len(self.tableWidget.selectedRanges()) > 0:
            self.item_selected(self.tableWidget.currentRow())
        else:
            makePNG(self.dpfa, 'dpfa')
            pixmap = QtGui.QPixmap("dpfa.png")
            self.draw_4.setPixmap(pixmap)

    def item_selected(self, row):
        item = self.tableWidget.item(row, 1).text()
        emphasizePNG(self.dpfa, item, 'dpfa')
        pixmap = QtGui.QPixmap("dpfa.png")
        self.draw_4.setPixmap(pixmap)
        
def makePNG(A, file_name):
    if type(A).__name__ == 'PFA':
        dot = Digraph(comment='PFA', format='png')
        # draw nodes
        for i in range(A.nbS):
            initial_prob = round(A.initial[i],2)
            final_prob = round(A.final[i],2)
            dot.node(str(i),'{} : {} : {}'.format(initial_prob,str(i),final_prob))
        # draw edges
        for alphabet in A.transitions.keys():
            for i in range(A.nbS):
                for j in range(A.nbS):
                    if A.transitions[alphabet][i,j] > 0:
                        probability = round(A.transitions[alphabet][i,j],2)
                        dot.edge(str(i), str(j), '{}, {}'.format(alphabet,probability))
        # draw start edge
        if A.nbS != 0:
            dot.attr('node',shape='none')
            dot.node('')
            dot.edge('','0')
        else:
            dot.node('Empty Automaton', shape='none')
        dot.render(file_name)

    elif type(A).__name__ == 'DFA':
        dot = Digraph(comment='DFA', format='png')
        # draw nodes
        for i in range(A.nbS):
            #initial_prob = round(A.initial[i],2)
            #final_prob = round(A.final[i],2)
            if A.final_states[i] == 1:
                dot.node(str(i),'{}'.format(str(i)), shape = 'doublecircle')
            elif A.final_states[i] == 0:
                dot.node(str(i),'{}'.format(str(i)), shape = 'circle')

        if A.nbL > 2:
            check_dict = {}
            replace_char = {} # dict {from : alphabet of one character}
            for from_state, character in A.transitions.keys():
                (from_state , to_state) = (from_state, A.transitions[(from_state, character)])
                if (from_state, to_state) in check_dict.keys():
                    check_dict[(from_state, to_state)][0] += 1
                    check_dict[(from_state, to_state)][1] = character
                else:
                    check_dict[(from_state, to_state)] = [1, character]
            for (from_state, to_state) in check_dict.keys():
                if check_dict[(from_state, to_state)][0] == 1:
                    replace_char[from_state] = check_dict[(from_state, to_state)][1]
            # draw edges
            for (from_state, to_state) in check_dict.keys():
                if check_dict[(from_state, to_state)][0] == 1:
                    dot.edge(str(from_state), str(to_state), replace_char[from_state])
                elif check_dict[(from_state, to_state)][0] == A.nbL - 1:
                    dot.edge(str(from_state), str(to_state), r'∑\\'+replace_char[from_state])
                elif check_dict[(from_state, to_state)][0] == A.nbL:
                    dot.edge(str(from_state), str(to_state), '∑')
        else:
            for from_state, character in A.transitions.keys():
                (from_state , to_state) = (from_state, A.transitions[(from_state, character)])
                dot.edge(str(from_state), str(to_state), character)
        # draw start edge
        dot.attr('node',shape='none')
        dot.node('')
        dot.edge('','0')
        dot.render(file_name)

def emphasizePNG(RA, string, file_name):
    edge_dict = {}
    dot = Digraph(comment='PFA', format='png')

    # draw nodes
    for i in range(RA.nbS):
        initial_prob = round(RA.initial[i],2)
        final_prob = round(RA.final[i],2)
        dot.node(str(i),'{} : {} : {}'.format(initial_prob,str(i),final_prob))

    str_idx = 0
    current_node = 0
    # color the start node
    initial_prob = round(RA.initial[current_node],2)
    final_prob = round(RA.final[current_node],2)
    dot.node(str(current_node), '{} : {} : {}'.format(initial_prob,current_node,final_prob),style='filled',fillcolor='yellow')
    
    while str_idx < len(string):
        next_node = np.nonzero(RA.transitions[string[str_idx]][current_node])[0][0]
        initial_prob = round(RA.initial[next_node],2)
        final_prob = round(RA.final[next_node],2)
        # color the nodes that matche string
        dot.node(str(next_node), '{} : {} : {}'.format(initial_prob,next_node,final_prob),style='filled',fillcolor='yellow')
        character = string[str_idx]
        probability = round(RA.transitions[character][current_node,next_node],2)
        # draw edges that matche string
        dot.edge(str(current_node), str(next_node), '{}, {}'.format(character,probability),style='bold')
        edge_dict[(current_node, next_node, character)] = None
        current_node = next_node
        str_idx += 1

    # draw remaining edges
    for alphabet in RA.transitions.keys():
        for i in range(RA.nbS):
            for j in range(RA.nbS):
                if (i,j, alphabet) not in edge_dict.keys() and RA.transitions[alphabet][i,j] > 0:
                    probability = round(RA.transitions[alphabet][i,j],2)
                    dot.edge(str(i), str(j), '{}, {}'.format(alphabet,probability),style='dashed')

    # start edge
    if RA.nbS != 0:
        dot.attr('node',shape='none')
        dot.node('')
        dot.edge('','0',style='bold')
        dot.render(file_name)
    else:
        dot.node('Empty Automaton', shape='none') 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()