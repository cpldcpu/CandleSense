#!/usr/bin/env python3
"""
High-performance real-time plotter for capacitive flicker sensor data using PyQtGraph
Usage: sudo make monitor | python3 plot_realtime_qt.py

Install dependencies:
pip3 install pyqtgraph PyQt5 numpy
"""

import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from collections import deque
import csv
from datetime import datetime

class FlickerSensorPlotter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Configure PyQtGraph for better performance
        pg.setConfigOptions(antialias=False)  # Disable antialiasing for speed
        pg.setConfigOption('background', 'k')  # Black background
        pg.setConfigOption('foreground', 'w')  # White foreground
        
        # Data storage
        self.max_points = 2000
        self.raw_data = deque(maxlen=self.max_points)
        self.avg_data = deque(maxlen=self.max_points)
        self.hp_data = deque(maxlen=self.max_points)
        self.zero_cross_data = deque(maxlen=self.max_points)
        self.timestamps = deque(maxlen=self.max_points)
        self.x_data = deque(maxlen=self.max_points)
        
        # Update control
        self.update_counter = 0
        self.update_interval = 32  # Update every 32 samples for even better performance
        
        # CSV logging
        self.csv_filename = f"Flicker_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'raw', 'avg', 'hp', 'zero_cross'])
        
        # Sample counter
        self.sample_count = 0
        
        self.setup_ui()
        self.setup_timer()
        
    def setup_ui(self):
        self.setWindowTitle('Capacitive Flicker Sensor - Real-time Data')
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Create plot widgets
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)
        
        # Create three subplots
        self.plot1 = self.plot_widget.addPlot(row=0, col=0, title="Raw & Filtered Signals")
        self.plot1.setLabel('left', 'ADC Value')
        self.plot1.addLegend()
        self.plot1.showGrid(x=True, y=True, alpha=0.3)
        
        self.plot2 = self.plot_widget.addPlot(row=1, col=0, title="Flicker Detection Signal")
        self.plot2.setLabel('left', 'Flicker Delta')
        self.plot2.addLegend()
        self.plot2.showGrid(x=True, y=True, alpha=0.3)
        self.plot2.addLine(y=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
        
        self.plot3 = self.plot_widget.addPlot(row=2, col=0, title="Zero Crossing Detection")
        self.plot3.setLabel('left', 'Zero Cross')
        self.plot3.setLabel('bottom', 'Samples')
        self.plot3.addLegend()
        self.plot3.showGrid(x=True, y=True, alpha=0.3)
        self.plot3.setYRange(-0.1, 1.1)  # Zero crossing is binary 0/1
        
        # Create plot curves
        self.raw_curve = self.plot1.plot(pen=pg.mkPen('c', width=1), name='Raw')
        self.avg_curve = self.plot1.plot(pen=pg.mkPen('y', width=1), name='Avg')
        self.hp_curve = self.plot2.plot(pen=pg.mkPen('g', width=1), name='Flicker Signal')
        self.zero_cross_curve = self.plot3.plot(pen=pg.mkPen('m', width=2), name='Zero Cross', symbol='o', symbolSize=4)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - waiting for data...")
        
    def setup_timer(self):
        # Timer to read from stdin
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.read_data)
        self.timer.start(1)  # Read every 1ms for maximum responsiveness
        
    def read_data(self):
        try:
            # Non-blocking read from stdin
            import select
            if select.select([sys.stdin], [], [], 0.0)[0]:
                line = sys.stdin.readline()
                if line.strip() and not line.startswith('Capacitive'):
                    parts = line.strip().split('\t')
                    if len(parts) == 4:
                        timestamp = datetime.now()
                        raw = int(parts[0])
                        avg = int(parts[1])
                        hp = int(parts[2])
                        zero_cross = int(parts[3])
                        
                        # Store data
                        self.sample_count += 1
                        self.timestamps.append(timestamp)
                        self.raw_data.append(raw)
                        self.avg_data.append(avg)
                        self.hp_data.append(hp)
                        self.zero_cross_data.append(zero_cross)
                        self.x_data.append(self.sample_count)
                        
                        # Log to CSV
                        self.csv_writer.writerow([timestamp.isoformat(), raw, avg, hp, zero_cross])
                        
                        # Update plots periodically
                        self.update_counter += 1
                        if self.update_counter >= self.update_interval:
                            self.update_plots()
                            self.update_counter = 0
                            
        except Exception as e:
            print(f"Error reading data: {e}", file=sys.stderr)
    
    def update_plots(self):
        if len(self.raw_data) < 2:
            return
            
        try:
            # Convert to numpy arrays for faster plotting
            x = np.array(self.x_data)
            raw = np.array(self.raw_data)
            avg = np.array(self.avg_data) >> 5  # Shift right by 5 to match your filtering
            hp = np.array(self.hp_data)
            zero_cross = np.array(self.zero_cross_data)
            
            # Update curves
            self.raw_curve.setData(x, raw)
            self.avg_curve.setData(x, avg)
            self.hp_curve.setData(x, hp)
            self.zero_cross_curve.setData(x, zero_cross)
            
            # Update status
            sample_rate = len(self.raw_data) / max(1, (self.timestamps[-1] - self.timestamps[0]).total_seconds()) if len(self.timestamps) > 1 else 0
            self.status_bar.showMessage(
                f"Samples: {self.sample_count} | Rate: {sample_rate:.1f} Hz | "
                f"Raw: {raw[-1]} | Flicker: {hp[-1]} | Zero Cross: {zero_cross[-1]} | File: {self.csv_filename}"
            )
            
        except Exception as e:
            print(f"Error updating plots: {e}", file=sys.stderr)
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.csv_file.close()
        print(f"\nData saved to {self.csv_filename}")
        print(f"Total samples collected: {self.sample_count}")
        event.accept()

def main():
    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)
    
    # Create and show the plotter
    plotter = FlickerSensorPlotter()
    plotter.show()
    
    try:
        # Run the application
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("\nStopping data collection...")
        plotter.close()

if __name__ == "__main__":
    main()
