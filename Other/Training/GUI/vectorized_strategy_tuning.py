# -*- coding: utf-8 -*-

from Other.Training.GUI import design
from Framework.Training.VectorizedStrategy import *
from View import plot_timeseries_vs_another

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from PyQt4 import QtCore, QtGui
import sys

from hashlib import sha1
from matplotlib import pyplot as plt
import os

class StratTuningApp(QtGui.QMainWindow, design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(StratTuningApp, self).__init__(parent)
        self.setupUi(self)
        
        #Initializes Combo boxes options
        self.init_combos ()
            
        #Connecting buttons with their respective functions
        self.connect_buttons ()        
        
        #init graphic objects
        self.init_graphics ()
        
        #Other objects not related to the GUI
        self.vs = VectorizedStrategy ()
        self.bIndicatorListInitialized = False
        
    def init_combos (self):
        #instrument selection combo       
        for instrument in full_instrument_list:
            self.ComboInstrument.addItem (instrument, instrument)
        
        #timeframes selection combo
        for tf in ['M15', 'H1', 'H4', 'D']:
            self.combo_fastTimeframe.addItem (tf)
            self.combo_slowTimeframe.addItem (tf)
        
        #criterium combo => might be deprecated soon
        for c in ['first', 'second', 'both']:
            self.comboBox_criterium.addItem (c)
        
    def connect_buttons (self):
        #loads an instrument
        self.btnInstrumentLoader.clicked.connect (self.load_instrument)
        
        #clear instruments
        self.btnInstrumentClear.clicked.connect (self.clear_cache)
        
        #runs the strategy
        self.btnStrategyRunner.clicked.connect (self.runStrategy)
        
        #saves the strategy
        self.pushButton_saveStrategy.clicked.connect (self.save_strategy_parameters)
        self.pushButton_loadStrategy.clicked.connect (self.load_strategy_parameters)
        
        #recreates instrument labels
        self.pushButton_relabel.clicked.connect (self.relabel)
        
        #visualize an indicator timeseries plot, histogram, other things to be added
        self.pushButton_visualize.clicked.connect (self.visualize)
        
    def init_graphics (self):
        self.canvas = None
        self.toolbar = None
        
        self.init_canvas (target=self.graphicsView_plot1)
        self.init_canvas (target=self.graphicsView_plot2)
        self.init_canvas (target=self.graphicsView_plot3)
        
    def init_canvas (self, fig = None, target = None):
        if fig is None:
            self.figure = plt.figure ()
        else:
            self.figure = fig
            
        self.canvas = FigureCanvas(self.figure)
        # use addToolbar to add toolbars to the main window directly!
        if self.toolbar is None:
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.addToolBar(self.toolbar)
        
        self.scene = QtGui.QGraphicsScene()
        self.scene.addWidget(self.canvas)
        
        if target is None:
            target = self.graphicsView_plot1
        target.setScene (self.scene)
        
    def clear_cache (self):
        del self.vs.dsh.ds_dict
        self.vs.dsh.ds_dict = {}
        del self.vs.preds_hash_table_dict
        self.vs.preds_hash_table_dict = {}

    #Actions triggered by the GUI buttons        
        
    def load_instrument (self):
        #deliberately adds 2y6m of data in order to account for lags when computing the features
        #not needed when features are loaded from the disk
        set_from_to_times(obj = self.vs, 
                          from_time = str(self.dateTimeEdit_from.dateTime().addYears(-2).addMonths(-6).toPyDateTime()),
                          to_time = str(self.dateTimeEdit_to.dateTime().toPyDateTime()),
                          )
        
        self.vs.load_instrument(instrument=str(self.ComboInstrument.currentText()),
                                timeframe = str (self.combo_fastTimeframe.currentText()),
                                other_timeframes = [str (self.combo_slowTimeframe.currentText())],
                                slow_timeframe_delay = self.spinBox_daily_delay.value() )
        
        #----------removes the extra 2y6m of data------------#
        set_from_to_times(obj = self.vs, 
                          from_time = str(self.dateTimeEdit_from.dateTime().toPyDateTime()),
                          to_time = str(self.dateTimeEdit_to.dateTime().toPyDateTime()),
                          )
        set_from_to_times(obj = self.vs.dsh, 
                          from_time = str(self.dateTimeEdit_from.dateTime().toPyDateTime()),
                          to_time = str(self.dateTimeEdit_to.dateTime().toPyDateTime()),
                          )
        
        #---------------------------------------------------#
        
        for key in self.vs.dsh.ds_dict.keys ():
            print (key)
        
        fig = plt.figure ()
        plt.plot (self.vs.ds.df.Close)
        
        self.init_canvas (fig)
                
        #init visualization combo box
        if not self.bIndicatorListInitialized:
            try:
                self.runStrategy () #needs to do this to have all indicators loaded
                for col in self.vs.ds.f_df.columns:
                    self.comboBox_indicatorToVisualize.addItem (col, col)
                    self.comboBox_versusIndicatorToVisualize.addItem (col, col)
                self.comboBox_indicatorToVisualize.addItem ('None')
                self.comboBox_versusIndicatorToVisualize.addItem ('None')
                self.bIndicatorListInitialized = True
            except:
                pass
            
# =============================================================================
#         self.comboBox_indicatorToVisualize.addItem ('None')
#             
#         for indicator in self.vs.ds.f_df.columns:
#             self.comboBox_indicatorToVisualize.addItem (indicator, indicator)
#             self.comboBox_versusIndicatorToVisualize.addItem (indicator, indicator)
#         
# =============================================================================
    def visualize (self):
        fig = plot_timeseries_vs_another (ts1 = self.vs.ds.f_df[str(self.comboBox_indicatorToVisualize.currentText())], 
                                    ts2 = self.vs.ds.f_df[str(self.comboBox_versusIndicatorToVisualize.currentText())], 
                   bSameAxis = False, 
                   figsize=(10,5),
                   label1 = str(self.comboBox_indicatorToVisualize.currentText()),
                   label2 = str(self.comboBox_versusIndicatorToVisualize.currentText()),
                   #y_bounds2 = (-1.1, 1.1),
                   bMultiple = False, 
                   bSave=False)                   
        
        self.init_canvas (fig, target = self.graphicsView_plot1)
        
        fig = plt.figure (figsize=(10,5))
        plt.title (str(self.comboBox_indicatorToVisualize.currentText()) + ' Histogram')
        plt.hist (self.vs.ds.f_df[str(self.comboBox_indicatorToVisualize.currentText())], bins=50)        
        self.init_canvas (fig, target = self.graphicsView_plot2)
        
        
    def relabel (self):
        ds = self.vs.ds
        
        ds.computeLabels (                         
                       bVaryStopTarget=True,
                       stop_fn = None, 
                       target_fn = None, 
                       target_multiple=self.doubleSpinBox_targetMultiple.value(),
                       min_stop = self.doubleSpinBox_minStop.value(),
                       vol_denominator = self.doubleSpinBox_volDenominator.value()                       
                       )
        
        fig = plot_timeseries_vs_another (ts1 = ds.f_df['Close' + '_' + ds.timeframe], 
                                    ts2 = ds.l_df.Labels, 
                   bSameAxis = False, 
                   figsize=(10,5),
                   y_bounds2 = (-1.1, 1.1),
                   bMultiple = False, 
                   bSave=False, 
                   title = 'Labels')
        
        self.init_canvas (fig, target = self.graphicsView_plot1)
        
        try:
            self.vs.preds_hash_table_dict [ds.ccy_pair] = sha1('Hello World').hexdigest ()
        except:
            pass
        
    def runStrategy (self):
        self.vs.hit_miss_cache.clear ()
        self.vs.init_pnl_dataframe ()
        kwargs = self.buildKwargsForStrategy ()
        
        self.vs.compute_pred_multiple (**kwargs)
        
        serial_gap = self.spinBox_serialGap.value()
        if serial_gap > 0:
            self.vs.ds.removeSerialPredictions (serial_gap)
        
        #scene = self.getScene(self.vs.plot_pnl())
        self.init_canvas(fig=self.vs.plot_multiple_pnl(bPlotSum = self.checkBox_plotPnLSum.isChecked ()), 
                         target=self.graphicsView_plot1)
        
        self.checkBox_plotPnLSum.stateChanged.connect (lambda x: self.init_canvas(fig=self.vs.plot_multiple_pnl(bPlotSum = self.checkBox_plotPnLSum.isChecked ()), 
                         target=self.graphicsView_plot1))
        
        self.init_canvas(fig=self.vs.plot_signals(), target=self.graphicsView_plot2)
        self.init_canvas(fig=self.vs.plot_hist(), target=self.graphicsView_plot3)
        
        self.plainTextEdit_output.appendPlainText(self.vs.summarize_stats ())
        
        
        
    def save_strategy_parameters (self):
        filename = str(self.comboBox_strategyFilename.currentText())
        
        if len (filename) > 3:        
            if filename.find ('.stp') < 0:
                filename += '.stp'
            
            f = open (os.path.join(strats_path, filename), 'w')
            f.write (str(self.buildKwargsForStrategy()))
            f.close ()
        
        
    def load_strategy_parameters (self):
        filename = str(self.comboBox_strategyFilename.currentText())
        
        if True:
            f = open (os.path.join(strats_path, filename), 'r')
            kwargs = eval(f.read ())
            print (str(kwargs))
            self.textEdit_function.setText (kwargs['func'])
            try:
                index = self.comboBox_criterium.findText(kwargs['criterium'], QtCore.Qt.MatchFixedString)
                self.comboBox_criterium.setCurrentIndex(index) 
            except:
                pass
            
            self.combo_fastTimeframe.setEditText (kwargs['fast_timeframe'])
            self.combo_slowTimeframe.setEditText (kwargs['slow_timeframe'])
            
            self.doubleSpinBox_rhoMin.setValue (kwargs['rho_min'])
            self.doubleSpinBox_rhoMax.setValue(kwargs['rho_max'])
            self.doubleSpinBox_residMin.setValue(kwargs['resid_min'])
            self.doubleSpinBox_residMax.setValue(kwargs['resid_max'])
            self.doubleSpinBox_rsiFastMin.setValue(kwargs['RSI_fast_min'])
            self.doubleSpinBox_rsiFastMax.setValue(kwargs['RSI_fast_max'])
            self.doubleSpinBox_rsiSlowMin.setValue(kwargs['RSI_slow_min'])
            self.doubleSpinBox_rsiSlowMax.setValue(kwargs['RSI_slow_max'])
            self.spinBox_halflifeMin.setValue(kwargs['halflife_min'])
            self.spinBox_halflifeMax.setValue(kwargs['halflife_max'])
            self.spinBox_netTrendlinesMin.setValue(kwargs['net_trendlines_min'])
            self.spinBox_netTrendlinesMax.setValue(kwargs['net_trendlines_max'])
            self.spinBox_trendlinesDeltaMin.setValue(kwargs['trendlines_delta_min'])
            self.spinBox_trendlinesDeltaMax.setValue(kwargs['trendlines_delta_max'])
            self.doubleSpinBox_closeOverMAmin.setValue(kwargs['close_over_ma_min'])
            self.doubleSpinBox_closeOverMAmax.setValue(kwargs['close_over_ma_max'])            
            self.checkBox_avoidOBSlow.setChecked (kwargs['avoid_overbought_slow'])
            self.checkBox_avoidOBFast.setChecked (kwargs['avoid_overbought_fast'])
            self.spinBox_OBSlowWindow.setValue(kwargs['overbought_slow_window'])
            self.spinBox_OBFastWindow.setValue(kwargs['overbought_fast_window'])            
            self.spinBox_serialGap.setValue(kwargs['serial_gap'])
        #except:
        #    print ('Error loading strategy')
        
    def buildKwargsForStrategy (self):
        kwargs = {}
        kwargs['func'] = str(self.textEdit_function.toPlainText())
        kwargs['criterium'] = str(self.comboBox_criterium.currentText())
        
        kwargs['fast_timeframe'] = str(self.combo_fastTimeframe.currentText())
        kwargs['slow_timeframe'] = str(self.combo_slowTimeframe.currentText())
        
        kwargs['rho_min'] = self.doubleSpinBox_rhoMin.value()
        kwargs['rho_max'] = self.doubleSpinBox_rhoMax.value()
        kwargs['resid_min'] = self.doubleSpinBox_residMin.value()
        kwargs['resid_max'] = self.doubleSpinBox_residMax.value()
        kwargs['RSI_fast_min'] = self.doubleSpinBox_rsiFastMin.value()
        kwargs['RSI_fast_max'] = self.doubleSpinBox_rsiFastMax.value()
        kwargs['RSI_slow_min'] = self.doubleSpinBox_rsiSlowMin.value()
        kwargs['RSI_slow_max'] = self.doubleSpinBox_rsiSlowMax.value()
        kwargs['halflife_min'] = self.spinBox_halflifeMin.value()
        kwargs['halflife_max'] = self.spinBox_halflifeMax.value()
        kwargs['net_trendlines_min'] = self.spinBox_netTrendlinesMin.value()
        kwargs['net_trendlines_max'] = self.spinBox_netTrendlinesMax.value()
        kwargs['trendlines_delta_min'] = self.spinBox_trendlinesDeltaMin.value()
        kwargs['trendlines_delta_max'] = self.spinBox_trendlinesDeltaMax.value()
        
        kwargs['close_over_ma_min'] = self.doubleSpinBox_closeOverMAmin.value()
        kwargs['close_over_ma_max'] = self.doubleSpinBox_closeOverMAmax.value()
        
        kwargs['avoid_overbought_slow'] = self.checkBox_avoidOBSlow.isChecked ()
        kwargs['avoid_overbought_fast'] = self.checkBox_avoidOBFast.isChecked ()
        kwargs['overbought_slow_window'] = self.spinBox_OBSlowWindow.value()
        kwargs['overbought_fast_window'] = self.spinBox_OBFastWindow.value()
        
        kwargs['serial_gap'] = self.spinBox_serialGap.value()
        
        kwargs['plot_pnl_sum'] = self.checkBox_plotPnLSum.isChecked ()
        
        return kwargs
        
    def getScene (self, fig):
        self.figure = fig        
        self.canvas.figure = self.figure
        
        # use addToolbar to add toolbars to the main window directly!
        #self.toolbar = NavigationToolbar(self.canvas, self)
        #self.addToolBar(self.toolbar)
        #scene = QtGui.QGraphicsScene()
        #layout = QtGui.QVBoxLayout()
        #layout.addWidget(self.canvas)
        #scene.addWidget(self.canvas)
        
        return self.scene
        
def main():
    app = QtGui.QApplication(sys.argv)
    form = StratTuningApp()
    form.show()
    app.exec_()
    
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    form = StratTuningApp()
    form.show()
    app.exec_()