import numpy as np
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class ECGBeatClassifier:
    def __init__(self, downsampled_ecg, detection_method_peaks_array, p=0.1, p_short=0.75, p_long=1.20) -> None:
        """
        The mean_RR interval is used to define thresholds for short (RR{<u}) and long (RR{>u}) RR intervals.
        The thresholds are defined by the user.

        Args:
            downsampled_ecg (np.array): The unfiltered ECG signal.
            detection_method_peaks_array (np.array): The indices of the detected R-peaks.
            p (float, optional): The percentage deviation allowed from the mean RR interval. Defaults to 0.1.
            p_short (float, optional): The percentage of the mean RR interval that defines the short RR interval threshold. Defaults to 0.75.
            p_long (float, optional): The percentage of the mean RR interval that defines the long RR interval threshold. Defaults to 1.20.
        """
        self.fs = 250  # <-- Sampling frequency (Hz) of the ECG signal
        self.downsampled_ecg = downsampled_ecg
        self.detection_method_peaks_array = detection_method_peaks_array

        # Convert the ECG signal indices to time (in seconds)
        self.time = np.arange(0, len(downsampled_ecg) / self.fs, 1/ self.fs)  
        self.detector_peak_times = detection_method_peaks_array / self.fs

        # mean and std of RR intervals (in seconds)
        self.rr_intervals = np.diff(self.detector_peak_times)
        self.mean_rr = np.mean(self.rr_intervals)
        self.std_rr = np.std(self.rr_intervals)
        
        # RR interval thresholds 
        self.p = p
        self.p_short = p_short
        self.short_rr_threshold = p_short * self.mean_rr
        self.p_long = p_long
        self.long_rr_threshold = p_long * self.mean_rr


    def beat_classify(self):
        """
        Classify the detected R-peaks into different beat types based on the RR intervals:
            1) Sinus-Sinus: RRi{u} and RRi-1{u}, where RR{u} means within 10% of mean_RR.
            2) Interrupted-Sinus: RRi-1{u} and RRi{<u}, where RR{<u} means less than or equal to 75% of mean_RR.
            3) PAC: RRi-1{<u} and RRi{u}, where (RRi-1{<u} + RRi{u})/2 > RR{u}. 
            4) PVC: RRi-1{<u} and RRi{>u}, where (RRi-1{<u} + RRi{>u})/2 = RR{u} and RR{>u} means greater than or equal to 120% of mean_RR.
            5) Post-Pause Sinus: RRi-1{>u} and RRi{u}.
            6) Bigeminy Sinus: RRi-1{>u} and RRi{<u}.
            7) Undetected R-peak: RRi-1{>u} or RRi{>u}, where  RRi-1{>u}/2 or RRi{>u}/2 = RR{u}.
            8) Misfire Detection: RRi-1{<u} and RRi{<u}, where RRi-1{<u} + RRi{<u} = RR{u}.

            maybes indicate correct pattern however below short or long thresholds for full classification:
            9) maybe PVC: RRi-1{<u} and RRi{>u}, in which either RRi-1{<u} or RRi{>u} is not within the defined thresholds. Otherwise, it would be classified as a PVC Beat.
            10) maybe PAC: RRi-1{<u} and RRi{u}, in which RRi-1{<u} is not within the defined threshold. Otherwise, it would be classified as a PAC Beat.
            11) maybe Interrupted-Sinus: RRi-1{u} and RRi{<u}, in which RRi{<u} is not within the defined threshold. Otherwise, it would be classified as a Interrupted-Sinus Beat.
            12) maybe Bigeminy Sinus: RRi-1{>u} and RRi{<u}, in which either RRi-1{>u} or RRi{<u} is not within the defined thresholds. Otherwise, it would be classified as a Bigeminy Sinus.
            13) maybe Post-Pause Sinus: RRi-1{>u} and RRi{u}, in which RRi{u} is not within the defined threshold. Otherwise, it would be classified as a Post-Pause Sinus Beat. 

            14) Other:
            1- RRi-1{u} and RRi{>u}, this can be a Sinus-Sinus that preceeds an Undetected Beat, but is not classified otherwise.
            2- RRi-1{>u} and RRi{>u}, this does not have a classification.

        Returns:
            df (pd.DataFrame): A DataFrame containing the detected R-peaks and their corresponding beat classifications.
        """
        # Set variables and thresholds to be used in the function
        downsampled_ecg = self.downsampled_ecg
        detector_peak_times = self.detector_peak_times
        detection_method_peaks_array = self.detection_method_peaks_array
        
        p = self.p
        long_rr_threshold = self.long_rr_threshold
        short_rr_threshold = self.short_rr_threshold
        rr_intervals, mean_rr = self.rr_intervals, self.mean_rr

        # Initialize lists beat types meeting Thresholds:
        sinus_sinus_x, sinus_sinus_y = [], []
        interrupted_sinus_x, interrupted_sinus_y = [], []
        pac_x, pac_y = [], []
        pvc_x, pvc_y = [], []
        post_pvc_x, post_pvc_y = [], []
        is_between_pvcs_x, is_between_pvcs_y = [], []

        missed_beat_x, missed_beat_y = [], []
        misfire_x, misfire_y = [], []

        # Initialize lists for beat types not meeting Thresholds:
        maybe_interrupted_sinus_x, maybe_interrupted_sinus_y = [], []
        maybe_pac_x, maybe_pac_y = [], []
        maybe_pvc_x, maybe_pvc_y = [], []
        maybe_post_pvc_x, maybe_post_pvc_y = [], []
        maybe_between_pvcs_x, maybe_between_pvcs_y = [], []
        
        # Unclassified beats:
        other_x, other_y = [], []
        
        for i in range(1, len(rr_intervals)):
            # Meeting Thresholds: 
            is_sinus_sinus = abs(mean_rr - rr_intervals[i-1]) <= (p * mean_rr) and abs(mean_rr - rr_intervals[i]) <= (p * mean_rr) 
            is_interrupted_sinus = abs(mean_rr - rr_intervals[i-1]) <= (p * mean_rr) and rr_intervals[i] <= short_rr_threshold 
            is_pac = (rr_intervals[i-1] <= short_rr_threshold and abs(mean_rr - rr_intervals[i]) <= (p * mean_rr)) and ((rr_intervals[i-1] + rr_intervals[i]) / 2) < ((1+p) * mean_rr)
            is_pvc = (rr_intervals[i-1] <= short_rr_threshold and rr_intervals[i] >= long_rr_threshold) and ((1-p) * mean_rr) <= ((rr_intervals[i-1] + rr_intervals[i]) / 2) <= ((1+p) * mean_rr)
            is_post_pvc = rr_intervals[i-1] >= long_rr_threshold and abs(mean_rr - rr_intervals[i]) <= (p * mean_rr)
            is_between_pvcs = rr_intervals[i-1] >= long_rr_threshold and rr_intervals[i] <= short_rr_threshold

            is_missed_beat = rr_intervals[i-1] >= long_rr_threshold and ((1-p) * mean_rr) <= (rr_intervals[i-1] / 2) <= ((1+p) * mean_rr) 
            is_misfire = rr_intervals[i-1] <= short_rr_threshold and rr_intervals[i] <= short_rr_threshold and ((1-p) * mean_rr) <= (rr_intervals[i-1] + rr_intervals[i]) <= ((1+p) * mean_rr)
            
            # Not Meeting Thresholds:
            maybe_interrupted_sinus = abs(mean_rr - rr_intervals[i-1]) <= (p * mean_rr) and rr_intervals[i] <= ((1-p) * mean_rr) 
            maybe_pac = rr_intervals[i-1] <= ((1-p) * mean_rr) and abs(mean_rr - rr_intervals[i]) <= (p * mean_rr) and ((rr_intervals[i-1] + rr_intervals[i]) / 2) < ((1+p) * mean_rr)
            maybe_pvc = (rr_intervals[i-1] <= ((1-p) * mean_rr) and rr_intervals[i] >= ((1+p) * mean_rr)) and ((1-p) * mean_rr) <= ((rr_intervals[i-1] + rr_intervals[i]) / 2) <= ((1+p) * mean_rr)
            maybe_post_pvc = rr_intervals[i-1] >= long_rr_threshold and abs(mean_rr - rr_intervals[i]) <= (p * mean_rr)
            maybe_between_pvcs = rr_intervals[i-1] >= ((1+p) * mean_rr) and rr_intervals[i] <= ((1-p) * mean_rr)
            
            if is_sinus_sinus:
                sinus_sinus_x.append(detector_peak_times[i])
                sinus_sinus_y.append(downsampled_ecg[detection_method_peaks_array[i]]) 
            elif is_interrupted_sinus:
                interrupted_sinus_x.append(detector_peak_times[i])
                interrupted_sinus_y.append(downsampled_ecg[detection_method_peaks_array[i]])
            elif maybe_interrupted_sinus and not is_interrupted_sinus:
                maybe_interrupted_sinus_x.append(detector_peak_times[i])
                maybe_interrupted_sinus_y.append(downsampled_ecg[detection_method_peaks_array[i]])
            elif is_pac and not is_pvc:
                pac_x.append(detector_peak_times[i])
                pac_y.append(downsampled_ecg[detection_method_peaks_array[i]])
            elif maybe_pac and not is_pac and not is_pvc:
                maybe_pac_x.append(detector_peak_times[i])
                maybe_pac_y.append(downsampled_ecg[detection_method_peaks_array[i]])
            elif is_pvc:
                pvc_x.append(detector_peak_times[i])
                pvc_y.append(downsampled_ecg[detection_method_peaks_array[i]])
            elif maybe_pvc and not is_pvc:
                maybe_pvc_x.append(detector_peak_times[i])
                maybe_pvc_y.append(downsampled_ecg[detection_method_peaks_array[i]])
            elif is_post_pvc and not is_missed_beat:
                post_pvc_x.append(detector_peak_times[i])
                post_pvc_y.append(downsampled_ecg[detection_method_peaks_array[i]])
            elif maybe_post_pvc and not is_post_pvc and not is_missed_beat:
                maybe_post_pvc_x.append(detector_peak_times[i])
                maybe_post_pvc_y.append(downsampled_ecg[detection_method_peaks_array[i]])
            elif is_between_pvcs:
                is_between_pvcs_x.append(detector_peak_times[i])
                is_between_pvcs_y.append(downsampled_ecg[detection_method_peaks_array[i]])
            elif maybe_between_pvcs and not is_between_pvcs:
                maybe_between_pvcs_x.append(detector_peak_times[i])
                maybe_between_pvcs_y.append(downsampled_ecg[detection_method_peaks_array[i]])
            elif is_missed_beat:
                missed_beat_x.append((detector_peak_times[i-1] + detector_peak_times[i]) / 2)
                missed_beat_y.append(downsampled_ecg[detection_method_peaks_array[i]])
                post_pvc_x.append(detector_peak_times[i])
                post_pvc_y.append(downsampled_ecg[detection_method_peaks_array[i]])
            elif is_misfire:
                misfire_x.append(detector_peak_times[i])
                misfire_y.append(downsampled_ecg[detection_method_peaks_array[i]])
            else:
                other_x.append(detector_peak_times[i])
                other_y.append(downsampled_ecg[detection_method_peaks_array[i]])

        # Create a dictionary where keys are column names and values are the lists
        data = {
            'sinus_sinus_x': sinus_sinus_x, 'sinus_sinus_y': sinus_sinus_y, 
            'interrupted_sinus_x': interrupted_sinus_x, 'interrupted_sinus_y': interrupted_sinus_y, 
            'maybe_interrupted_sinus_x': maybe_interrupted_sinus_x, 'maybe_interrupted_sinus_y': maybe_interrupted_sinus_y, 
            'pac_x': pac_x, 'pac_y': pac_y, 
            'maybe_pac_x': maybe_pac_x, 'maybe_pac_y': maybe_pac_y, 
            'pvc_x': pvc_x, 'pvc_y': pvc_y, 
            'maybe_pvc_x': maybe_pvc_x, 'maybe_pvc_y': maybe_pvc_y, 
            'post_pvc_x': post_pvc_x, 'post_pvc_y': post_pvc_y, 
            'maybe_post_pvc_x': maybe_post_pvc_x, 'maybe_post_pvc_y': maybe_post_pvc_y, 
            'is_between_pvcs_x': is_between_pvcs_x, 'is_between_pvcs_y': is_between_pvcs_y, 
            'maybe_between_pvcs_x': maybe_between_pvcs_x, 'maybe_between_pvcs_y': maybe_between_pvcs_y, 
            'missed_beat_x': missed_beat_x, 'missed_beat_y': missed_beat_y, 
            'misfire_x': misfire_x, 'misfire_y': misfire_y, 
            'other_x': other_x, 'other_y': other_y
        }

        # Pad shorter lists with np.nan so they don't cause errors when creating a DataFrame or plotting
        max_length = max(len(lst) for lst in data.values())
        for key, value in data.items():
            if len(value) < max_length:
                data[key] = value + [np.nan] * (max_length - len(value))

        # Create a DataFrame and drop columns where all values are np.nan
        df = pd.DataFrame(data)
        df = df.dropna(axis=1, how='all')
        return df


    def beat_breakdown(self,show_plot=False):
        """
        Count the number of beats in each beat classification category. Plot beat classification percentages.

        Returns:
            beat_percentages (dict): A dictionary containing the percentage of each beat classification category.
            total_beat_count (int): The total number of beats detected.
            mean_rr (float): The mean RR interval in milliseconds.
            std_rr (float): The standard deviation of the RR intervals in milliseconds.
        """
        # Call the beat_classify function to classify the beats then count them 
        df = self.beat_classify()

        # Create Scatter objects and traces for each beat classification
        trace_info = [
            {'name': 'Sinus-Sinus', 'color': 'rgba(0, 100, 0, 1)', 'x': 'sinus_sinus_x'},  # darkgreen
            {'name': 'Interrupted-Sinus', 'color': 'rgba(255, 215, 0, 1)', 'x': 'interrupted_sinus_x'},  # gold
            {'name': 'maybe Interrupted-Sinus', 'color': 'rgba(255, 250, 205, 1)', 'x': 'maybe_interrupted_sinus_x'},  # lemonchiffon
            {'name': 'PAC', 'color': 'rgba(147, 112, 219, 1)', 'x': 'pac_x', 'y': 'pac_y'},  # mediumpurple
            {'name': 'maybe PAC', 'color': 'rgba(216, 191, 216, 1)', 'x': 'maybe_pac_x'},  # thistle
            {'name': 'PVC', 'color': 'rgba(65, 105, 225, 1)', 'x': 'pvc_x'},  # royalblue
            {'name': 'maybe PVC', 'color': 'rgba(176, 196, 222, 1)', 'x': 'maybe_pvc_x'},  # lightsteelblue
            {'name': 'Post-Pause Sinus', 'color': 'rgba(255, 140, 0, 1)', 'x': 'post_pvc_x'},  # darkorange
            {'name': 'maybe Post-Pause Sinus', 'color': 'rgba(255, 228, 181, 1)', 'x': 'maybe_post_pvc_x'},  # moccasin
            {'name': 'Bigeminy Sinus', 'color': 'rgba(0, 0, 139, 1)', 'x': 'is_between_pvcs_x'},  # darkblue
            {'name': 'Misfire Detections', 'color': 'rgba(101, 67, 33, 1)', 'x': 'misfire_x'},  # mud_brown
            {'name': 'Other', 'color': 'rgba(0, 255, 255, 1)', 'x': 'other_x'},  # cyan
            {'name': 'maybe Bigeminy Sinus', 'color': 'rgba(0, 0, 139, 0.5)', 'x': 'maybe_between_pvcs_x'},  # darkblue (0.5 transparency)
            {'name': 'Undetected R-peaks', 'color': 'rgb(255, 0, 0,1)', 'x': 'missed_beat_x'}  # red
            ]

        # Calculate the total count of all 'x' keys excluding 'Undetected R-peaks'. Add 2 to account for the starting and ending RRs not classified
        total_beat_count = sum(len(df[trace['x']].dropna()) if trace['x'] in df.columns and trace['name'] != 'Undetected R-peaks' else 0 for trace in trace_info) + 2
        undetected_beat_count = sum(len(df[trace['x']].dropna()) if trace['x'] in df.columns and trace['name'] == 'Undetected R-peaks' else 0 for trace in trace_info)
        
        fig = go.Figure()
        beat_percentages = {}

        # For each beat type, calculate the percent of it from the total beat count and barplot it
        for trace in trace_info:
            if trace['x'] in df.columns:
                if trace['name'] == 'Undetected R-peaks': beat_percentage = 105
                else: beat_percentage = (len(df[trace['x']].dropna()) / total_beat_count) * 100
                fig.add_trace(go.Bar( x=[trace['name']], y=[round(beat_percentage, 2)], name=trace['name'], marker_color=trace['color'], width=0.4))

                # Add beat percentage to overall beat percentages dictionary
                beat_percentages[trace['name']] = round(beat_percentage, 2) if beat_percentage != 105 else 105

        fig.update_layout(
            width=800,  # width of the figure
            height=600,  # height of the figure
            showlegend=False,  # hide the legend
            plot_bgcolor='rgba(240, 240, 240, 0.3)',  # light grey
            paper_bgcolor='rgba(255, 255, 255, 1)',  # white
            
            title=f'Percent Breakdown of Beats (n={total_beat_count}), Undetected R-peaks (n={undetected_beat_count})',
            
            xaxis_title='Beat Classifcation',
            xaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='black',
                showgrid=True,
                gridwidth=0.1,
                gridcolor='rgba(255, 255, 255, 1)'),
            
            yaxis_title='Percent of Total Beats Detected (%)',
            yaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='black',
                showgrid=True,
                gridwidth=0.1,
                gridcolor='rgba(255, 255, 255, 1)'))
        
        if(show_plot): fig.show()
        return beat_percentages, total_beat_count, self.mean_rr*1000, self.std_rr*1000


    def plot_classified_ecg(self, testcase, annotations_array=None, method_title='R-peak', plot_breakdown=False, HTML_OUTPUT=False):
        """
        Interactive plot of ECG signal with the detected and annotated R-peaks, and the beat classifications.

        Args:
            testcase (str): The name of the test case.
            annotations_array (np.array, optional): The indices of the annotated R-peaks. Defaults to None.
            method_title (str, optional): The method_title used to detect the R-peaks. Defaults to 'R-peak'.
            HTML_OUTPUT (bool, optional): Whether to save the plot as an HTML file. Defaults to False.
        """
        # Set variables and thresholds to be used in the function
        downsampled_ecg = self.downsampled_ecg
        detector_peak_times = self.detector_peak_times
        detection_method_peaks_array = self.detection_method_peaks_array

        time = self.time
        mean_rr, std_rr = self.mean_rr, self.std_rr 
        long_rr_threshold = self.long_rr_threshold
        short_rr_threshold = self.short_rr_threshold
        p, p_short, p_long = self.p, self.p_short, self.p_long

        # Call the beat_classify function to classify the beats
        df = self.beat_classify()
        if (plot_breakdown): df_percent, total_beat_count, _, _ = self.beat_breakdown(show_plot=True)
        else: df_percent, total_beat_count, _, _ = self.beat_breakdown()

        # Create a plotly figure
        fig = go.Figure()

        # Add ECG signal trace
        fig.add_trace(go.Scatter(x=time, y=downsampled_ecg, mode='lines', name='ECG Signal', line=dict(color='rgba(128, 128, 128, 0.5)'))) # light blue

        # Add annotated R-peaks trace as vertical dashed-lines, if available
        if annotations_array is not None:
            annotated_peak_times = annotations_array / self.fs
            x_values = []
            y_values = []
            for peak_time in annotated_peak_times:
                x_values.extend([peak_time, peak_time, None])  # Add None to break the line
                y_values.extend([min(downsampled_ecg), max(downsampled_ecg), None])  # Add None to break the line
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines',
                name='Annotated R-peaks',
                line=dict(color='rgba(0, 0, 0, 1)', dash='dash', width=2),
                showlegend=True  # Ensure it appears in the legend
            ))
                        
        # # Add annotated R-peaks trace as dots, if available
        # if annotations_array is not None:
        #     annotated_peak_times = annotations_array / self.fs
        #     fig.add_trace(go.Scatter(x=annotated_peak_times, y=downsampled_ecg[annotations_array], 
        #                              mode='markers', name='Annotated R-peaks', marker=dict(color='rgba(128, 128, 128, 1)'))) # grey

        # Add detected R-peaks trace as dots
        fig.add_trace(go.Scatter(x=detector_peak_times, y=downsampled_ecg[detection_method_peaks_array], 
                                 mode='markers', name=f'Detected R-peaks({total_beat_count})', marker=dict(color='rgba(0, 0, 0, 1)'))) # black

        # Create Scatter objects and traces for each beat classification
        trace_info = [
            {'name': 'Sinus-Sinus', 'color': 'rgba(0, 100, 0, 1)', 'x': 'sinus_sinus_x', 'y': 'sinus_sinus_y'},  # darkgreen
            {'name': 'Interrupted-Sinus', 'color': 'rgba(255, 215, 0, 1)', 'x': 'interrupted_sinus_x', 'y': 'interrupted_sinus_y'},  # gold
            {'name': 'maybe Interrupted-Sinus', 'color': 'rgba(255, 250, 205, 1)', 'x': 'maybe_interrupted_sinus_x', 'y': 'maybe_interrupted_sinus_y'},  # lemonchiffon
            {'name': 'PAC', 'color': 'rgba(147, 112, 219, 1)', 'x': 'pac_x', 'y': 'pac_y'},  # mediumpurple
            {'name': 'maybe PAC', 'color': 'rgba(216, 191, 216, 1)', 'x': 'maybe_pac_x', 'y': 'maybe_pac_y'},  # thistle
            {'name': 'PVC', 'color': 'rgba(65, 105, 225, 1)', 'x': 'pvc_x', 'y': 'pvc_y'},  # royalblue
            {'name': 'maybe PVC', 'color': 'rgba(176, 196, 222, 1)', 'x': 'maybe_pvc_x', 'y': 'maybe_pvc_y'},  # lightsteelblue
            {'name': 'Post-Pause Sinus', 'color': 'rgba(255, 140, 0, 1)', 'x': 'post_pvc_x', 'y': 'post_pvc_y'},  # darkorange
            {'name': 'maybe Post-Pause Sinus', 'color': 'rgba(255, 228, 181, 1)', 'x': 'maybe_post_pvc_x', 'y': 'maybe_post_pvc_y'},  # moccasin
            {'name': 'Bigeminy Sinus', 'color': 'rgba(0, 0, 139, 1)', 'x': 'is_between_pvcs_x', 'y': 'is_between_pvcs_y'},  # darkblue
            {'name': 'maybe Bigeminy Sinus', 'color': 'rgba(0, 0, 139, 0.5)', 'x': 'maybe_between_pvcs_x', 'y': 'maybe_between_pvcs_y'},  # darkblue (0.5 transparency)
            {'name': 'Misfire Detections', 'color': 'rgba(101, 67, 33, 1)', 'x': 'misfire_x', 'y': 'misfire_y'},  # mud_brown
            {'name': 'Other', 'color': 'rgba(0, 255, 255, 1)', 'x': 'other_x', 'y': 'other_y'},  # cyan
            {'name': 'Undetected R-peaks', 'color': 'rgb(255, 0, 0,1)', 'x': 'missed_beat_x', 'y': 'missed_beat_y'}  # red
            ]

        traces = []
        for info in trace_info:
            x = df[info['x']] if info['x'] in df.columns else []
            y = df[info['y']] if info['y'] in df.columns else []
            name_percent = df_percent.get(info['name'])
            trace = go.Scatter(mode='markers', name=f"{info['name']}({name_percent}%)", marker=dict(color=info['color']), x=x, y=y)
            fig.add_trace(trace)
            traces.append(trace)


        # Update figure layout
        fig.update_layout(
            plot_bgcolor='rgba(240, 240, 240, 0.3)',  # light grey
            paper_bgcolor='rgba(255, 255, 255, 1)',  # white

            title=f"""{method_title} Beat Classification for: {testcase}
            <br>\tMean-HR = {60/mean_rr:.2f}bpm ({mean_rr*1000:.2f}ms Mean-RR), \tHR Variation = {std_rr/mean_rr*100:.2f}% of Mean-HR 
            <br>\tShort-RR Threshold <= {short_rr_threshold*1000:.2f}ms ({p_short*100:.2f}% of Mean-RR), \tLong-RR Threshold >= {long_rr_threshold*1000:.2f}ms ({p_long*100:.2f}% of Mean-RR)""", 
            
            xaxis_title='Time (seconds)',
            xaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='black',
                showgrid=True,
                gridwidth=0.1,
                gridcolor='rgba(255, 255, 255, 1)'),
                
            yaxis_title='ECG Signal (&mu;V)',
            yaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='black',
                showgrid=True,
                gridwidth=0.1,
                gridcolor='rgba(255, 255, 255, 1)'),
            
            legend=dict(
                yanchor="top",
                xanchor="left",))

        fig.show()

        # Save the plot as an HTML file
        if(HTML_OUTPUT): pyo.plot(fig, filename= f'./html/{testcase}_{method_title}_beat_classification.html', auto_open=False)


class ComparisonPlotter:
    def __init__(self, 
                 downsampled_ecg2=None, ECG2_on=False,
                 downsampled_ecg3=None, ECG3_on=False,
                 ECG2detections_array=None, ECG2Trig_on=False,
                 ECG3detections_array=None, ECG3Trig_on=False,
                 ECG2hamilton_array=None, ECG2hamilton_on=False,
                 ECG3hamilton_array=None, ECG3hamilton_on=False,
                 pulse_detections_array=None, PPGTrig_on=False,
                 annotations_array=None, annotations_on=False,
                 ECG2cnn_detector_peaks_array=None, CNN2_on=False,
                 ECG3cnn_detector_peaks_array=None, CNN3_on=False) -> None:    
        """
        Initialize the ComparisonPlotter class.

        Args:
            downsampled_ecg2 (np.array): The ECG signal from the second ECG lead.
            downsampled_ecg3 (np.array): The ECG signal from the third ECG lead.
            CNN (np.array): The indices of the detected R-peaks by the CNN detector.
            ECG2cnn_detector_peaks_array (np.array): The indices of the detected R-peaks by the ECG2 detector.
            ECG3cnn_detector_peaks_array (np.array): The indices of the detected R-peaks by the ECG3 detector.
            ECG2detections_array (np.array): The indices of the detected R-peaks by the ECG2 detector.
            ECG3detections_array (np.array): The indices of the detected R-peaks by the ECG3 detector.
            VCG2detections_array (np.array): The indices of the detected R-peaks by the VCG2 detector.
            VCG3detections_array (np.array): The indices of the detected R-peaks by the VCG3 detector.
            pulse_detections_array (np.array): The indices of the detected PPG peaks.
            annotations_array (np.array): The indices of the annotated R-peaks.
        """
        self.fs = 250  # Sampling rate (Hz) of the ECG signal

        if ECG2_on: self.ECG2 = downsampled_ecg2
        else: self.ECG2 = None

        if ECG3_on: self.ECG3 = downsampled_ecg3
        else: self.ECG3 = None
        
        if CNN2_on: self.CNN2 = ECG2cnn_detector_peaks_array
        else: self.CNN2 = None
        
        if CNN3_on: self.CNN3 = ECG3cnn_detector_peaks_array
        else: self.CNN3 = None
        
        if ECG2Trig_on: self.ECG2Trig = ECG2detections_array
        else: self.ECG2Trig = None
        
        if ECG3Trig_on: self.ECG3Trig = ECG3detections_array
        else: self.ECG3Trig = None
        
        if ECG2hamilton_on: self.ECG2hamilton = ECG2hamilton_array
        else: self.ECG2hamilton = None
        
        if ECG3hamilton_on: self.ECG3hamilton = ECG3hamilton_array
        else: self.ECG3hamilton = None
        
        if PPGTrig_on: self.PPGTrig = pulse_detections_array
        else: self.PPGTrig = None
        
        if annotations_on: self.annotations = annotations_array
        else: self.annotations = None

        # Set the start at time 0, end time as max of the two signals
        self.PlotStart = 0
        if self.ECG2 is not None and self.ECG3 is not None: self.PlotEnd = max(len(self.ECG2), len(self.ECG3)) / self.fs
        elif self.ECG2 is not None and self.ECG3 is None: self.PlotEnd = len(self.ECG2) / self.fs
        elif self.ECG2 is None and self.ECG3 is not None: self.PlotEnd = len(self.ECG3) / self.fs


    def compare(self, PlotStart=None, PlotEnd=None, numbers=True, title='R-peak Detections'):
        """
        Plot the ECG signal with the detected R-peaks and the annotated R-peaks

        Args:
            PlotStart (int, optional): The starting time in seconds to plot the ECG signal. Defaults to 0.
            PlotEnd (int, optional): The ending time in seconds to plot the ECG signal. Defaults to max of the signal.
            numbers (bool, optional): Whether to show the numbers of the detected R-peaks. Defaults to True.
            title (str, optional): The title of the plot. Defaults to 'R-peak Detections'.
        """
        plt.figure(figsize=(20, 2.5))
        if PlotStart == None: PlotStart = self.PlotStart
        if PlotEnd == None: PlotEnd = self.PlotEnd

        # Plot the ECG signals
        # if self.ECG2 is not None: plt.plot(self.ECG2, color='lightblue', label='ECG2')
        if self.ECG2 is not None: plt.plot(self.ECG2, label='ECG')
        if self.ECG3 is not None: plt.plot(self.ECG3, label='ECG') # default color is blue

        # Plot PPG Triggered R-peaks
        if self.PPGTrig is not None:
            first_ppg = True
            for i, idx in enumerate(self.PPGTrig):
                if PlotStart <= idx / self.fs <= PlotEnd:
                    if self.ECG2 is not None and self.ECG3 is None:
                        plt.scatter(idx, self.ECG2[idx], color='midnightblue', marker='D', s=60, label='scanner gating' if first_ppg else "", zorder=3)
                        if numbers: plt.text(idx, self.ECG2[idx], str(i), fontsize=12, color='midnightblue', ha='right')
                    elif self.ECG2 is None and self.ECG3 is not None:
                        plt.scatter(idx, self.ECG3[idx], color='midnightblue', marker='D', s=60, label='scanner gating' if first_ppg else "", zorder=3)
                        if numbers: plt.text(idx, self.ECG3[idx], str(i), fontsize=12, color='midnightblue', ha='right')
                    else:
                        plt.scatter(idx, self.ECG2[idx], color='midnightblue', marker='D', s=60, label='scanner gating' if first_ppg else "", zorder=3)
                        plt.scatter(idx, self.ECG3[idx], color='midnightblue', marker='D', zorder=3)
                        if numbers:
                            plt.text(idx, self.ECG2[idx], str(i), fontsize=12, color='midnightblue', ha='right')
                            plt.text(idx, self.ECG3[idx], str(i), fontsize=12, color='midnightblue', ha='right')
                    first_ppg = False

        # Plot ECG2 Triggered R-peaks
        if self.ECG2Trig is not None:
            first_vcg = True
            for i, idx in enumerate(self.ECG2Trig):
                if PlotStart <= idx / self.fs <= PlotEnd:
                    if self.ECG2 is not None and self.ECG3 is None:
                        plt.scatter(idx, self.ECG2[idx], color='midnightblue', marker='D', s=60, label='scanner gating' if first_vcg else "", zorder=3)
                        if numbers: plt.text(idx, self.ECG2[idx], str(i), fontsize=12, color='midnightblue', ha='right')
                    elif self.ECG2 is None and self.ECG3 is not None:
                        plt.scatter(idx, self.ECG3[idx], color='midnightblue', marker='D', s=60, label='scanner gating' if first_vcg else "", zorder=3)
                        if numbers: plt.text(idx, self.ECG3[idx], str(i), fontsize=12, color='midnightblue', ha='right')
                    else:
                        plt.scatter(idx, self.ECG2[idx], color='midnightblue', marker='D', s=60, label='scanner gating' if first_vcg else "", zorder=3)
                        plt.scatter(idx, self.ECG3[idx], color='midnightblue', marker='D', zorder=3)
                        if numbers:
                            plt.text(idx, self.ECG2[idx], str(i), fontsize=12, color='midnightblue', ha='right')
                            plt.text(idx, self.ECG3[idx], str(i), fontsize=12, color='midnightblue', ha='right')
                    first_vcg = False

        # Plot ECG3 Triggered R-peaks
        if self.ECG3Trig is not None:
            first_vcg3 = True
            for i, idx in enumerate(self.ECG3Trig):
                if PlotStart <= idx / self.fs <= PlotEnd:
                    if self.ECG2 is not None and self.ECG3 is None:
                        plt.scatter(idx, self.ECG2[idx], color='red', marker='o', s=60, label='scanner gating' if first_vcg3 else "", zorder=3)
                        if numbers: plt.text(idx, self.ECG2[idx], str(i), fontsize=12, color='red', ha='right')
                    elif self.ECG2 is None and self.ECG3 is not None:
                        plt.scatter(idx, self.ECG3[idx], color='red', marker='o', s=60, label='scanner gating' if first_vcg3 else "", zorder=3)
                        if numbers: plt.text(idx, self.ECG3[idx], str(i), fontsize=12, color='red', ha='right')
                    else:
                        plt.scatter(idx, self.ECG2[idx], color='red', marker='o', s=60, label='scanner gating' if first_vcg3 else "", zorder=3)
                        plt.scatter(idx, self.ECG3[idx], color='red', marker='o', zorder=3)
                        if numbers:
                            plt.text(idx, self.ECG2[idx], str(i), fontsize=12, color='red', ha='right')
                            plt.text(idx, self.ECG3[idx], str(i), fontsize=12, color='red', ha='right')
                    first_vcg3 = False

        # Plot Hamilton2 Triggered R-peaks
        if self.ECG2hamilton is not None:
            first_hamilton2 = True
            for i, idx in enumerate(self.ECG2hamilton):
                if PlotStart <= idx / self.fs <= PlotEnd:
                    if self.ECG2 is not None and self.ECG3 is None:
                        plt.scatter(idx, self.ECG2[idx], color='darkorange', marker='v', s=60, label='Hamilton' if first_hamilton2 else "", zorder=3)
                        if numbers: plt.text(idx, self.ECG2[idx], str(i), fontsize=12, color='darkorange', ha='right')
                    elif self.ECG2 is None and self.ECG3 is not None:
                        plt.scatter(idx, self.ECG3[idx], color='darkorange', marker='v', s=60, label='Hamilton' if first_hamilton2 else "", zorder=3)
                        if numbers: plt.text(idx, self.ECG3[idx], str(i), fontsize=12, color='darkorange', ha='right')
                    else:
                        plt.scatter(idx, self.ECG2[idx], color='darkorange', marker='v', s=60, label='Hamilton' if first_hamilton2 else "", zorder=3)
                        plt.scatter(idx, self.ECG3[idx], color='darkorange', marker='v', zorder=3)
                        if numbers:
                            plt.text(idx, self.ECG2[idx], str(i), fontsize=12, color='darkorange', ha='right')
                            plt.text(idx, self.ECG3[idx], str(i), fontsize=12, color='darkorange', ha='right')
                    first_hamilton2 = False

        # Plot Hamilton3 Triggered R-peaks
        if self.ECG3hamilton is not None:
            first_hamilton3 = True
            for i, idx in enumerate(self.ECG3hamilton):
                if PlotStart <= idx / self.fs <= PlotEnd:
                    if self.ECG2 is not None and self.ECG3 is None:
                        plt.scatter(idx, self.ECG2[idx], color='darkorange', marker='v', s=60, label='Hamilton' if first_hamilton3 else "", zorder=3)
                        if numbers: plt.text(idx, self.ECG2[idx], str(i), fontsize=12, color='darkorange', ha='right')
                    elif self.ECG2 is None and self.ECG3 is not None:
                        plt.scatter(idx, self.ECG3[idx], color='darkorange', marker='v', s=60, label='Hamilton' if first_hamilton3 else "", zorder=3)
                        if numbers: plt.text(idx, self.ECG3[idx], str(i), fontsize=12, color='darkorange', ha='right')
                    else:
                        plt.scatter(idx, self.ECG2[idx], color='darkorange', marker='v', s=60, label='Hamilton' if first_hamilton3 else "", zorder=3)
                        plt.scatter(idx, self.ECG3[idx], color='darkorange', marker='v', zorder=3)
                        if numbers:
                            plt.text(idx, self.ECG2[idx], str(i), fontsize=12, color='darkorange', ha='right')
                            plt.text(idx, self.ECG3[idx], str(i), fontsize=12, color='darkorange', ha='right')
                    first_hamilton3 = False

        # Plot CNN2 Detector R-peaks for ECG2
        if self.CNN2 is not None:
            first_cnn2 = True
            for i, idx in enumerate(self.CNN2):
                if PlotStart <= idx / self.fs <= PlotEnd:
                    if self.ECG2 is not None and self.ECG3 is None:
                        plt.scatter(idx, self.ECG2[idx], color='red', marker='o', s=60, label='retrospective gating' if first_cnn2 else "", zorder=3)
                        if numbers: plt.text(idx, self.ECG2[idx], str(i), fontsize=12, color='red', ha='right')
                    elif self.ECG2 is None and self.ECG3 is not None:
                        plt.scatter(idx, self.ECG3[idx], color='red', marker='o', s=60, label='retrospective gating' if first_cnn2 else "", zorder=3)
                        if numbers: plt.text(idx, self.ECG3[idx], str(i), fontsize=12, color='red', ha='right')
                    else:
                        plt.scatter(idx, self.ECG2[idx], color='red', marker='o', s=60, label='retrospective gating' if first_cnn2 else "", zorder=3)
                        plt.scatter(idx, self.ECG3[idx], color='red', marker='o', zorder=3)
                        if numbers:
                            plt.text(idx, self.ECG2[idx], str(i), fontsize=12, color='red', ha='right')
                            plt.text(idx, self.ECG3[idx], str(i), fontsize=12, color='red', ha='right')
                    first_cnn2 = False

        # Plot CNN3 Detector R-peaks for ECG3
        if self.CNN3 is not None:
            first_cnn3 = True
            for i, idx in enumerate(self.CNN3):
                if PlotStart <= idx / self.fs <= PlotEnd:
                    if self.ECG2 is not None and self.ECG3 is None:
                        plt.scatter(idx, self.ECG2[idx], color='red', marker='o', s=60, label='retrospective gating' if first_cnn3 else "", zorder=3)
                        if numbers: plt.text(idx, self.ECG2[idx], str(i), fontsize=12, color='red', ha='right')
                    elif self.ECG2 is None and self.ECG3 is not None:
                        plt.scatter(idx, self.ECG3[idx], color='red', marker='o', s=60, label='retrospective gating' if first_cnn3 else "", zorder=3)
                        if numbers: plt.text(idx, self.ECG3[idx], str(i), fontsize=12, color='red', ha='right')
                    else:
                        plt.scatter(idx, self.ECG2[idx], color='red', marker='o', s=60, label='retrospective gating' if first_cnn3 else "", zorder=3)
                        plt.scatter(idx, self.ECG3[idx], color='red', marker='o', zorder=3)
                        if numbers:
                            plt.text(idx, self.ECG2[idx], str(i), fontsize=12, color='red', ha='right')
                            plt.text(idx, self.ECG3[idx], str(i), fontsize=12, color='red', ha='right')
                    first_cnn3 = False

        # Plot Annotated R-peaks
        if self.annotations is not None:
            for ind, annotation in enumerate(self.annotations):
                if ind == 0: plt.axvline(x=annotation, linestyle='--', color='green', label='annotated labels', zorder=3)
                else: plt.axvline(x=annotation, linestyle='--', color='green', zorder=3)

        # Set x-ticks to appear every 250 data points within the x-axis range (1 second intervals)
        ax = plt.gca()
        ax.set_xlim([PlotStart * (self.fs - 1), PlotEnd * (self.fs + 1)])  
        x_ticks = np.arange(PlotStart * self.fs, (PlotEnd * self.fs) + 1, self.fs)  # Add 1 to include last tick value
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(lambda x, pos: f"{int((x - PlotStart * self.fs) / self.fs)}s")  # Adjust formatter to account for start time
        
        # Set y-ticks to and y-tick labels
        ax.set_ylim([-15000, 25000])  
        ax.set_yticklabels([])

        # Add vertical lines for each second
        if self.ECG2 is not None:
            for seconds_lines in range(0, len(self.ECG2), self.fs): plt.axvline(x=seconds_lines, linestyle=':', color='gray')
        else:
            for seconds_lines in range(0, len(self.ECG3), self.fs): plt.axvline(x=seconds_lines, linestyle=':', color='gray')

        # Add facecolor to the and remove gridlines
        ax.grid(False)
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

        # Add title, labels, and legend to the plot
        plt.margins(0.01)
        plt.title(f"{title}", fontsize=20, color='black')
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.xlabel('Time (seconds)', fontsize=15, color='black')
        # plt.ylabel('ECG Signal (µV)', fontsize=15, color='black')
        # plt.legend(bbox_to_anchor=(0, 1.03), loc="lower left", borderaxespad=0, ncol=5, fontsize=18)
        plt.show()
