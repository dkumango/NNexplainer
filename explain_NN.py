################################################################################
# NNexplainer
# Intepretation of predictions for neural network models
# This code needs optimization
# 2024-10-02
################################################################################
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Input
from keras import Model
from keras import utils
from keras import initializers

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
import numpy as np

## BEGIN CLASS ####################################################################
class NNexplainer():   
    def __init__(self, random_state=1, scaler=None):
        self._random_state = random_state
        self.scaler = scaler
        self._model = None
        self._feature_names = None
        self._task = None            # regression or classification
        self._fmin = []              # min value of each feature
        self._fmax = []              # max value of each feature

    # build a neural network model
    def build_NN(self, X_train, y_train, params, task='regression'):

        # min max of each feature
        for i in range(len(X_train.columns)):
            self._fmin.append(X_train.iloc[:,i].min())
            self._fmax.append(X_train.iloc[:,i].max())

        epochs = params['epochs']  
        batch_size = params['batch_size']  
        lr = params['lr']
        self._feature_names = params['feature_names']
        self._task = task
        X_train = self.scaler.transform(X_train)

        # Set the seed using keras.utils.set_random_seed. This will set:
        # 1) `numpy` seed
        # 2) backend random seed
        # 3) `python` random seed
        utils.set_random_seed(self._random_state)
        initializer = initializers.he_normal()              # GlorotNormal(), Ones()  
        last_layer = len(params['layers'])-1
        activator = 'leaky_relu'                            #  'relu'

        self._model = Sequential()
        self._model.add(Input(shape=(X_train.shape[1],)))
        for i in range(0,len(params['layers'])-1): 
            self._model.add(Dense(params['layers'][i], activation=activator, kernel_initializer=initializer))
        if task == 'regression':   # regression    
            self._model.add(Dense(params['layers'][last_layer], kernel_initializer=initializer))
        else:                      # classification
           self._model.add(Dense(params['layers'][last_layer], activation='softmax', kernel_initializer=initializer))

        #model.summary()  # show model structure
        if task == 'regression':
            loss='mse'
            metrics=['mae']
        else:     # classification
            loss='categorical_crossentropy'     
            metrics=['accuracy']
   
        # Compile model
        adam = optimizers.Adam(learning_rate=lr)
        self._model.compile(loss=loss,  
                    optimizer=adam, 
                    metrics=metrics)

        # model fitting (learning)
        disp = self._model.fit(X_train, y_train, 
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,        # print fitting process  
                        validation_split = 0.2)

        #return(model,disp)

        return(disp)
    
    # explain the model (regression/classification) ##################################
    def explainer(self, input):

        WEIGHTS = []
        layers = len(self._model.layers)
        for layer in self._model.layers: 
            #print(layer.get_config()['name'], layer.get_weights())
            WEIGHTS.append(layer.get_weights())

        inputs = [input]                           # input of each layer 
        for ln in range(layers):
            input_org = input[np.newaxis]
            extractor = Model(inputs=self._model.inputs,
                            outputs= self._model.layers[ln].output) 
            
            input_new = extractor(input_org)[0].numpy()
            inputs.append(input_new)   
        
        cont_matrix_before = np.diag(input)                   # contribution matrix of each layer

        for ln in range(layers):
            #cont_matrix_this = []    
            nodes = WEIGHTS[ln][0].shape[1]                  # number of output nodes in the layer

            wi_xi = np.matmul(np.array(WEIGHTS[ln][0]).T, np.array(cont_matrix_before))
            cont_matrix_this = wi_xi

            for i in range(nodes): 
                if (ln == (layers-1)) & (self._task == 'classification'):  
                    TR = 1
                else:            
                    tmp = pd.Series(inputs[ln])*pd.Series(WEIGHTS[ln][0][:,i])
                    if tmp.sum() == 0:
                        TR = 0
                    else:
                        TR = inputs[ln+1][i]/tmp.sum()
                
                cont_matrix_this[i] *= TR 

            cont_matrix_before = cont_matrix_this
            #print('cont_matrix_before\n', cont_matrix_before)    
        if self._task == 'regression':
            cont_list = cont_matrix_before[0]
        else:            # classification 
            cont_list = cont_matrix_before    
        print(cont_list)                
        return(cont_list)

    # plot the contribution of each feature (Regression) ####################################
    def plot_contribution_R(self, input):
        no_feature = len(self._feature_names)

        input_scaled = self.scaler.transform(input.to_frame().T)
        prediction = self._model.predict(input_scaled)[0][0]
        cont_list= self.explainer(input_scaled[0])
        cont_list = np.array(cont_list)

        max_lim = cont_list.max()
        min_lim = cont_list.min()

        f_name = []
        for j in range(no_feature): 
            f_name.append(self._feature_names[j] + ' =' + repr(input.iloc[j]))

        bar_colors = ['salmon' if x < 0 else 'steelblue' for x in cont_list]

        title_text = 'Contribution Plot\n ( predict: ' + str(prediction) + ' )'
        plt.barh(f_name, cont_list, color=bar_colors)

        plt.xlim([min_lim,max_lim])
        plt.title(title_text)    
        plt.tight_layout()

        plt.show()
        return( pd.Series(cont_list, index=self._feature_names))

    # plot the contribution of each feature (Classification) ####################################
    def plot_contribution_C(self, input):
        no_feature = len(self._feature_names)

        input_scaled = self.scaler.transform(input.to_frame().T)
        prediction = self._model.predict(input_scaled)[0]
        prediction = 'calss'+str(np.argmax(prediction))
        cont_list= self.explainer(input_scaled[0])
        cont_list = np.array(cont_list)


        f_name = []
        for j in range(no_feature): 
            f_name.append(self._feature_names[j] + ' =' + repr(input.iloc[j]))

##
        data = cont_list

        data_shape = np.shape(data)

        # Take negative and positive data apart and cumulate
        def get_cumulated_array(data, **kwargs):
            cum = data.clip(**kwargs)
            cum = np.cumsum(cum, axis=0)
            d = np.zeros(np.shape(data))
            d[1:] = cum[:-1]
            return d  

        cumulated_data = get_cumulated_array(data, min=0)
        cumulated_data_neg = get_cumulated_array(data, max=0)

        # Re-merge negative and positive data.
        row_mask = (data < 0)
        cumulated_data[row_mask] = cumulated_data_neg[row_mask]
        data_stack = cumulated_data

        # class color
        cols = ["orange", "limegreen", "lightsteelblue", "mediumpurple", "violet", 
                "lightcoral", "gold", "skyblue", "gray", "brown"]

        title_text = 'Contribution Plot\n ( predict: ' + str(prediction) + ' )'

        fig, (ax, ax_table) = plt.subplots(2, 1, height_ratios=[2, 1])

        for i in np.arange(0, data_shape[0]):
            #ax.bar(np.arange(data_shape[1]), data[i], bottom=data_stack[i], color=cols[i],)
            #ax.barh(np.arange(data_shape[1]), data[i],  left=data_stack[i], color=cols[i],)
            ax.barh(f_name, data[i],  left=data_stack[i], color=cols[i], label='class'+str(i))
        
        # vertical line indicating the 0 value
        ax.plot([0, 0], [-0.5, no_feature+0.5], color='gray', linewidth=0.8)    

        ax.set_title(title_text) 
        # Shrink current axis by 20%
        box = ax.get_position()
        fig.subplots_adjust(right= 0.7)

        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='lower left', bbox_to_anchor=(1, 0.5))

        # generate data table
        col_labels = ['Opposite','Support','Total']
        row_labels = []
        for i in range(data_shape[0]):
            row_labels.append('class'+str(i))

        table_vals = []
        for i in range(data_shape[0]):
            tmp = np.array(data[i,:]).round(2) 
            table_vals.append([tmp[tmp<0].sum().round(2), 
                            tmp[tmp>0].sum().round(2), tmp.sum().round(2)])

        row_colors = cols[0:data_shape[0]]

        #plotting
        ax_table.table(cellText=table_vals,
                    colWidths=[0.25] * data_shape[0],
                    rowLabels=row_labels,
                    colLabels=col_labels,
                    rowColours=row_colors,
                    loc='upper right')

        ax_table.set_axis_off()

        plt.tight_layout()
        plt.show()
##
        return( pd.DataFrame(cont_list, columns=self._feature_names))


    # plot the contribution of each feature (interactive, regression) ####################################
    def plot_contribution_interactive_R(self, input_org):

        #----
        input_scaled = self.scaler.transform(input_org.to_frame().T)
        prediction = self._model.predict(input_scaled)[0][0]
        cont_list= self.explainer(input_scaled[0])
        features = self._feature_names
        #----

        x = features
        y = cont_list

        fig = plt.figure()
        fig.set_figwidth(10)

        ax_bar = fig.add_subplot(111)         # hbar chart

        # Adjust the subplots region to leave some space for the sliders and buttons
        fig.subplots_adjust(left=0.1, right= 0.5, bottom=0.1)

        # Draw the initial plot
        bar_colors = ['salmon' if x < 0 else 'steelblue' for x in y]
        title_text = 'Contribution Plot\n ( predict: ' + str(prediction) + ' )'
        ax_bar.barh(x, y, color=bar_colors)

        ax_bar.title.set_text(title_text)    
        #plt.tight_layout()


        # Add sliders for tweaking the parameters
        # Define an axes area and draw a slider in it
        sliders =[]
        for i in range(len(features)):
            smin = self._fmin[i]
            smax = self._fmax[i]

            sobj_ax = fig.add_axes([0.58, 0.12 + i*0.055, 0.25, 0.06], facecolor='yellow')  # [left, bottom, width, height]
            sobj = Slider(sobj_ax, features[i], smin, smax, valinit=input_org.iloc[i], color='limegreen')

            sliders.append(sobj)


        def sliders_on_changed(val):
            ax_bar.clear()
        
            #----
            input_changed = input_org.copy()
            for i in range(len(sliders)):
                input_changed.iloc[i] =sliders[i].val
            
            input_scaled = self.scaler.transform(input_changed.to_frame().T)
            prediction = self._model.predict(input_scaled)[0][0]
            cont_list= self.explainer(input_scaled[0])
            
            bar_colors = ['salmon' if x < 0 else 'steelblue' for x in cont_list]
            title_text = 'Contribution Plot\n ( predict: ' + str(prediction) + ' )'
            ax_bar.barh(features, cont_list, color=bar_colors)
            ax_bar.title.set_text(title_text)
            #----    
            # update the hbar plot
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        for s in sliders:
            s.on_changed(sliders_on_changed)

        # Add a button for resetting the parameters
        reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
        reset_button = Button(reset_button_ax, 'Reset', color='yellow', hovercolor='0.975')
        def reset_button_on_clicked(mouse_event):
            for s in sliders:
                s.reset()

        reset_button.on_clicked(reset_button_on_clicked)

        #plt.tight_layout()
        plt.show()
        return( pd.Series(cont_list, index=self._feature_names))


    # plot the contribution of each feature (interactive, classificaton) ####################################
    def plot_contribution_interactive_C(self, input_org):

        #----
        no_feature = len(self._feature_names)

        input_scaled = self.scaler.transform(input_org.to_frame().T)
        prediction = self._model.predict(input_scaled)[0]
        prediction = 'calss'+str(np.argmax(prediction))
        cont_list= self.explainer(input_scaled[0])
        cont_list = np.array(cont_list)
        #----

##
        data = cont_list

        data_shape = np.shape(data)

        # Take negative and positive data apart and cumulate
        def get_cumulated_array(data, **kwargs):
            cum = data.clip(**kwargs)
            cum = np.cumsum(cum, axis=0)
            d = np.zeros(np.shape(data))
            d[1:] = cum[:-1]
            return d  

        cumulated_data = get_cumulated_array(data, min=0)
        cumulated_data_neg = get_cumulated_array(data, max=0)

        # Re-merge negative and positive data.
        row_mask = (data < 0)
        cumulated_data[row_mask] = cumulated_data_neg[row_mask]
        data_stack = cumulated_data

        # class color
        cols = ["orange", "limegreen", "lightsteelblue", "mediumpurple", "violet", 
                "lightcoral", "gold", "skyblue", "gray", "brown"]

        title_text = 'Contribution Plot\n ( predict: ' + str(prediction) + ' )'

        fig, (ax, ax_table) = plt.subplots(2, 1, height_ratios=[2, 1])
        fig.set_figwidth(10)


        for i in np.arange(0, data_shape[0]):
            ax.barh(self._feature_names, data[i],  left=data_stack[i], color=cols[i], label='class'+str(i))
        
        # vertical line indicating the 0 value
        ax.plot([0, 0], [-0.5, no_feature+0.5], color='gray', linewidth=0.8)    

        ax.set_title(title_text) 
        # Shrink current axis by 20%
        box = ax.get_position()
        fig.subplots_adjust(right= 0.7)

        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        # ax.legend(loc='lower left', bbox_to_anchor=(1, 0.5))

        # generate data table
        col_labels = ['Opposite','Support','Total']
        row_labels = []
        for i in range(data_shape[0]):
            row_labels.append('class'+str(i))

        table_vals = []
        for i in range(data_shape[0]):
            tmp = np.array(data[i,:]).round(2) 
            table_vals.append([tmp[tmp<0].sum().round(2), 
                            tmp[tmp>0].sum().round(2), tmp.sum().round(2)])

        row_colors = cols[0:data_shape[0]]

        #plotting
        ax_table.table(cellText=table_vals,
                    colWidths=[0.25] * data_shape[0],
                    rowLabels=row_labels,
                    colLabels=col_labels,
                    rowColours=row_colors,
                    loc='upper right')

        ax_table.set_axis_off()

        plt.tight_layout()
        # Add sliders for tweaking the parameters
        # Define an axes area and draw a slider in it
        sliders =[]
        for i in range(no_feature):
            smin = self._fmin[i]
            smax = self._fmax[i]

            sobj_ax = fig.add_axes([0.6, 0.16 + i*0.055, 0.25, 0.06], facecolor='yellow')  # [left, bottom, width, height]
            sobj = Slider(sobj_ax, self._feature_names[i], smin, smax, valinit=input_org.iloc[i], color='tomato')

            sliders.append(sobj)


        def sliders_on_changed(val):   # 여기 고쳐야
            ax.clear()
            ax_table.clear()
        
            #----
            input_changed = input_org.copy()
            for i in range(len(sliders)):
                input_changed.iloc[i] =sliders[i].val
            
            input_scaled = self.scaler.transform(input_changed.to_frame().T)
            prediction = self._model.predict(input_scaled)[0]
            prediction = 'calss'+str(np.argmax(prediction))
            cont_list= self.explainer(input_scaled[0])
            cont_list = np.array(cont_list)

            data = cont_list
            data_shape = np.shape(data)

            # Take negative and positive data apart and cumulate
            cumulated_data = get_cumulated_array(data, min=0)
            cumulated_data_neg = get_cumulated_array(data, max=0)

            # Re-merge negative and positive data.
            row_mask = (data < 0)
            cumulated_data[row_mask] = cumulated_data_neg[row_mask]
            data_stack = cumulated_data

            title_text = 'Contribution Plot\n ( predict: ' + str(prediction) + ' )'

            for i in np.arange(0, data_shape[0]):
                ax.barh(self._feature_names, data[i],  left=data_stack[i], color=cols[i], label='class'+str(i))
            
            # vertical line indicating the 0 value
            ax.plot([0, 0], [-0.5, no_feature+0.5], color='gray', linewidth=0.8)    
            ax.set_title(title_text) 

            # redraw data table
            table_vals = []
            for i in range(data_shape[0]):
                tmp = np.array(data[i,:]).round(2) 
                table_vals.append([tmp[tmp<0].sum().round(2), 
                                tmp[tmp>0].sum().round(2), tmp.sum().round(2)])

            #plotting
            ax_table.table(cellText=table_vals,
                        colWidths=[0.25] * data_shape[0],
                        rowLabels=row_labels,
                        colLabels=col_labels,
                        rowColours=row_colors,
                        loc='upper right')
            ax_table.set_axis_off()

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        for s in sliders:
            s.on_changed(sliders_on_changed)

        # Add a button for resetting the parameters
        reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
        reset_button = Button(reset_button_ax, 'Reset', color='yellow', hovercolor='0.975')
        def reset_button_on_clicked(mouse_event):
            for s in sliders:
                s.reset()

        reset_button.on_clicked(reset_button_on_clicked)

        # Adjust the subplots region to leave some space for the sliders and buttons
        fig.subplots_adjust(left=0.1, right= 0.5, bottom=0.1)

        #plt.tight_layout()
        plt.show()
        return( pd.DataFrame(cont_list, columns=self._feature_names))




    # find max min ##################################################
    def _remove_outliers_and_find_min_max(self, X):
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = np.percentile(X, 25)
        Q3 = np.percentile(X, 75)
        
        # Calculate IQR
        IQR = Q3 - Q1
        
        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out the outliers
        filtered_X = [x for x in X if lower_bound <= x <= upper_bound]
        
        # Find min and max values in the filtered list
        min_value = min(filtered_X)
        max_value = max(filtered_X)
        
        return min_value, max_value

    # feature importance based on permutation test
    def plot_feature_importance(self, X_data, y_data):
        # baseline mae
        X_data_scaled = self.scaler.transform(X_data)
        base_score = self._model.evaluate(X_data_scaled, y_data, verbose=0)[1]

        # permutaion test 
        print('Permutation test.....')
        perm_scores = []
        for i in range(10):
            this_score = []
            for f in range(len(self._feature_names)):
                X_data_perm = X_data_scaled.copy()
                X_data_perm[:,f] = np.random.RandomState(seed=i).permutation(X_data_scaled[:,f])
                this_score.append(self._model.evaluate(X_data_perm, y_data, verbose=0)[1])

            perm_scores.append(this_score)

        perm_scores = pd.DataFrame(perm_scores, columns=self._feature_names)
        if self._task == 'regression':
            diff = perm_scores.mean(axis=0) - base_score
        else:
            diff = base_score - perm_scores.mean(axis=0)  

        df_sorted = diff.sort_values()
 
        plt.barh(df_sorted.index, df_sorted)
        plt.title('Feature Importance')
        plt.show()

       

## END CLASS #######################################################################