from mininet.net import Mininet
from mininet.node import Controller
from mininet.link import TCLink
from mininet.clean import cleanup
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from itertools import combinations
import time
class NetworkManager:
    def _init_(self,initial_hosts=4):
        self.initial_hosts=initial_hosts
        self.current_hosts=initial_hosts
        self.net=None
        self.hosts=[]
        self.historical_data=pd.DataFrame()
        self.model=None
        self.scaler=StandardScaler()
        self.dynamic_hosts=[]
        self.host_counter=initial_hosts
        self.all_possible_hosts=[f'h{i}' for i in range(1,8)]
    def create_network(self,new_hosts=0):
        cleanup()
        self.net=Mininet(controller=Controller,link=TCLink)
        total_hosts=self.current_hosts+new_hosts
        for i in range(1,total_hosts+1):
            host_name=f'h{i}'
            ip_address=f'10.0.0.{i}'
            host=self.net.addHost(host_name,ip=ip_address)
            self.hosts.append(host)
        num_switches=max(2,total_hosts//3)
        switches=[]
        for i in range(num_switches):
            switch=self.net.addSwitch(f's{i+1}')
            switches.append(switch)
        for host in self.hosts:
            switch=random.choice(switches)
            self.net.addLink(host,switch)
        for s1,s2 in combinations(switches,2):
            if random.random()<0.7:
                self.net.addLink(s1,s2)
        c0=self.net.addController('c0')
        self.net.start()
        self.current_hosts+=new_hosts
        return self.hosts
    def add_dynamic_host(self):
        self.host_counter+=1
        host_name=f'h{self.host_counter}'
        ip_address=f'10.0.0.{self.host_counter}'
        new_host=self.net.addHost(host_name,ip=ip_address)
        switches=[node for node in self.net.values() if isinstance(node,self.net.switch)]
        switch=random.choice(switches)
        self.net.addLink(new_host,switch)
        self.hosts.append(new_host)
        self.dynamic_hosts.append(new_host)
        if host_name not in self.all_possible_hosts:
            self.all_possible_hosts.append(host_name)
        return new_host
    def generate_network_data(self,sample_size=2000):
        all_host_names=self.all_possible_hosts
        slices_bandwidth={
            "gaming":{"bandwidth":400,"latency_range":(5,20)},
            "mobile_broadband":{"bandwidth":300,"latency_range":(10,30)},
            "iot":{"bandwidth":200,"latency_range":(15,40)},
            "industrial_automation":{"bandwidth":100,"latency_range":(5,15)}
        }
        host_slices={host:random.choice(list(slices_bandwidth.keys())) for host in all_host_names}
        data=[]
        peak_hours=range(9,18)
        for _ in range(sample_size):
            source_host=random.choice(all_host_names)
            dest_host=random.choice([h for h in all_host_names if h!=source_host])
            slice_name=host_slices[source_host]
            slice_config=slices_bandwidth[slice_name]
            base_bandwidth=slice_config["bandwidth"]
            latency_min,latency_max=slice_config["latency_range"]
            time_of_day=random.randint(0,23)
            is_peak_hour=time_of_day in peak_hours
            latency=round(random.uniform(latency_min,latency_max),2)
            congestion_level=round(random.uniform(20 if not is_peak_hour else 40,60 if not is_peak_hour else 80),2)
            link_quality=round(random.uniform(75,95),2)
            signal_strength=round(random.uniform(-65,-35),2)
            is_dynamic=(int(source_host[1])>=5) or (source_host in [h.name for h in self.dynamic_hosts])
            if is_dynamic:
                mobility_factor=random.uniform(0.8,1.2)
                latency*=random.uniform(1.05,1.2)
                link_quality*=random.uniform(0.9,1.0)
            else:
                mobility_factor=1.0
            bandwidth=round(base_bandwidth*(1-congestion_level/100*0.5)(link_quality/100)*mobility_factor(20/(latency+10)),2)
            jitter=round(5+(latency*0.1)+(congestion_level*0.1),2)
            packet_loss=round(max(0,min(5,(congestion_level/100)*3+(100-link_quality)/20)),2)
            throughput=round(bandwidth*(1-packet_loss/100),2)
            data.append({
                "source_host":source_host,
                "dest_host":dest_host,
                "slice_name":slice_name,
                "Bandwidth (Mbps)":bandwidth,
                "Latency (ms)":latency,
                "Packet Loss (%)":packet_loss,
                "Jitter (ms)":jitter,
                "Throughput (Mbps)":throughput,
                "Congestion Level (%)":congestion_level,
                "Link Quality (%)":link_quality,
                "Signal Strength (dBm)":signal_strength,
                "Time_of_day":time_of_day,
                "Is_Dynamic_Host":is_dynamic
            })
        df=pd.DataFrame(data)
        self.historical_data=pd.concat([self.historical_data,df],ignore_index=True)
        return df
    def prepare_data_for_cnn(self):
        features=self.historical_data[['Latency (ms)','Packet Loss (%)','Jitter (ms)','Congestion Level (%)','Link Quality (%)','Signal Strength (dBm)','Time_of_day']].values
        target=self.historical_data['Bandwidth (Mbps)'].values
        features_scaled=self.scaler.fit_transform(features)
        features_reshaped=features_scaled.reshape(features_scaled.shape[0],1,features_scaled.shape[1])
        return train_test_split(features_reshaped,target,test_size=0.2,random_state=42)
    def calculate_prediction_accuracy(self, X, y_true):
        """Calculate accuracy directly from predictions"""
        y_pred = self.model.predict(X, verbose=0)
        return np.mean(y_pred.flatten() / y_true) * 100 
    def build_and_train_cnn(self):
        X_train,X_test,y_train,y_test=self.prepare_data_for_cnn()
        model=Sequential([
            Conv1D(128,kernel_size=1,activation='relu',input_shape=(1,7)),
            BatchNormalization(),
            Conv1D(64,kernel_size=1,activation='relu'),
            BatchNormalization(),
            Flatten(),
            Dense(128,activation='relu'),
            Dropout(0.3),
            BatchNormalization(),
            Dense(64,activation='relu'),
            Dropout(0.2),
            BatchNormalization(),
            Dense(32,activation='relu'),
            Dropout(0.1),
            Dense(1,activation='linear')
        ])
        optimizer=Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer,loss='mse',metrics=['mae'])
        callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
        history=model.fit(X_train,y_train,epochs=100,batch_size=32,validation_data=(X_test,y_test),callbacks=[callback],verbose=1)
        model.compile
        self.model=model
        train_accuracies = []
        val_accuracies = []
        
        print("\nCalculating prediction accuracies...")
        for epoch in range(len(history.history['loss'])):
            train_acc = self.calculate_prediction_accuracy(X_train, y_train)
            val_acc = self.calculate_prediction_accuracy(X_test, y_test)
            
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
        
        # Add accuracies to history object
        history.history['accuracy'] = train_accuracies
        history.history['val_accuracy'] = val_accuracies
        
        # Print final accuracy
        print(f"\nFinal Training Accuracy: {train_accuracies[-1]:.2f}%")
        print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
        return history,X_test,y_test
    def generate_and_predict_new_host_data(self):
        new_host=self.add_dynamic_host()
        new_host_name=new_host.name
        if self.historical_data.empty:
            print("No historical data found. Generating initial data...")
            self.generate_network_data(sample_size=2000)
        host_data=self.historical_data[self.historical_data['source_host'].str.contains(f'h{self.host_counter}')]
        if host_data.empty:
            print(f"No historical data found for h{self.host_counter}. Please ensure dataset includes this host.")
            return pd.DataFrame()
        validation_data=[]
        grouped_data=host_data.groupby('dest_host').agg({
            'Bandwidth (Mbps)':'mean',
            'Latency (ms)':'mean',
            'Packet Loss (%)':'mean',
            'Jitter (ms)':'mean',
            'Congestion Level (%)':'mean',
            'Link Quality (%)':'mean',
            'Signal Strength (dBm)':'mean',
            'Time_of_day':'mean',
            'slice_name':'first'
        }).reset_index()
        for _,row in grouped_data.iterrows():
            features=row[['Latency (ms)','Packet Loss (%)','Jitter (ms)','Congestion Level (%)','Link Quality (%)','Signal Strength (dBm)','Time_of_day']].values.reshape(1,-1)
            scaled_features=self.scaler.transform(features)
            reshaped_features=scaled_features.reshape(scaled_features.shape[0],1,scaled_features.shape[1])
            predicted_bandwidth=self.model.predict(reshaped_features,verbose=0)[0][0]
            actual_bandwidth=row['Bandwidth (Mbps)']
            validation_data.append({
                'source':new_host_name,
                'destination':row['dest_host'],
                'slice_name':row['slice_name'],
                'actual_bandwidth':actual_bandwidth,
                'predicted_bandwidth':predicted_bandwidth,
                'error':abs(actual_bandwidth-predicted_bandwidth),
                'error_percentage':abs(actual_bandwidth-predicted_bandwidth)/actual_bandwidth*100
            })
        validation_df=pd.DataFrame(validation_data)
        print("\nDetailed Validation Results:")
        for _,row in validation_df.iterrows():
            print(f"\nConnection: {row['source']} → {row['destination']}")
            print(f"Service Slice: {row['slice_name']}")
            print(f"Actual Bandwidth: {row['actual_bandwidth']:.2f} Mbps")
            print(f"Predicted Bandwidth: {row['predicted_bandwidth']:.2f} Mbps")
            print(f"Error: {row['error']:.2f} Mbps ({row['error_percentage']:.2f}%)")
        return validation_df
    def stop_network(self):
        if self.net:
            self.net.stop()
        cleanup()
    def plot_prediction_comparison(self,validation_results):
        plt.figure(figsize=(12,8))
        plt.subplot(2,1,1)
        destinations=validation_results['destination']
        actual=validation_results['actual_bandwidth']
        predicted=validation_results['predicted_bandwidth']
        x=range(len(destinations))
        width=0.35
        plt.bar([i-width/2 for i in x],actual,width,label='Actual Bandwidth',color='blue',alpha=0.7)
        plt.bar([i+width/2 for i in x],predicted,width,label='Predicted Bandwidth',color='red',alpha=0.7)
        plt.xlabel('Destination Hosts')
        plt.ylabel('Bandwidth (Mbps)')
        plt.title('Actual vs Predicted Bandwidth for Dynamic Host')
        plt.xticks(x,destinations)
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(destinations,validation_results['error_percentage'],'g-o',label='Prediction Error %')
        plt.xlabel('Destination Hosts')
        plt.ylabel('Error Percentage (%)')
        plt.title('Prediction Error by Destination')
        plt.legend()
        plt.tight_layout()
        plt.savefig('bandwidth_prediction_analysis.png')
        plt.close()
def main():
    network_manager=NetworkManager(initial_hosts=4)
    network_manager.create_network()
    print("Generating initial network data...")
    initial_data=network_manager.generate_network_data(sample_size=2000)
    print("\nData Statistics:")
    print("Number of unique source hosts:",initial_data['source_host'].nunique())
    print("Number of unique destination hosts:",initial_data['dest_host'].nunique())
    print("Number of samples generated:",len(initial_data))
    print("\nAdding dynamic hosts and generating their data...")
    for _ in range(2):
        network_manager.add_dynamic_host()
        network_manager.generate_network_data(sample_size=500)
        time.sleep(1)
    print("\nBuilding and training the CNN model...")
    history,X_test,y_test=network_manager.build_and_train_cnn()
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Prediction Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_accuracy.png')
    plt.close()

    # Plot training history - MAE
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Training and Validation Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_mae.png')
    plt.close()
    print("\nGenerating and predicting bandwidth for a new dynamic host...")
    validation_results=network_manager.generate_and_predict_new_host_data()
    print("\nCalculating average bandwidths for all hosts...")
    host_averages=network_manager.historical_data.groupby('source_host')['Bandwidth (Mbps)'].mean().round(2)
    print("\nAverage Actual Bandwidth by Host:")
    print(host_averages)
    if not validation_results.empty:
        prediction_averages=validation_results.groupby('source').agg({
            'actual_bandwidth':'mean',
            'predicted_bandwidth':'mean',
            'error_percentage':'mean'
        }).round(2)
        print("\nPrediction Results Summary:")
        print(prediction_averages)
        plt.figure(figsize=(12,6))
        hosts=prediction_averages.index
        x=range(len(hosts))
        width=0.35
        plt.bar([i-width/2 for i in x],prediction_averages['actual_bandwidth'],width,label='Average Actual',color='blue',alpha=0.7)
        plt.bar([i+width/2 for i in x],prediction_averages['predicted_bandwidth'],width,label='Average Predicted',color='red',alpha=0.7)
        plt.xlabel('Hosts')
        plt.ylabel('Bandwidth (Mbps)')
        plt.title('Average Actual vs Predicted Bandwidth by Host')
        plt.xticks(x,hosts)
        plt.legend()
        for i in x:
                plt.text(i - width/2, prediction_averages['actual_bandwidth'].iloc[i],
                        f'{prediction_averages["actual_bandwidth"].iloc[i]:.1f}',
                        ha='center', va='bottom')
                plt.text(i + width/2, prediction_averages['predicted_bandwidth'].iloc[i],
                        f'{prediction_averages["predicted_bandwidth"].iloc[i]:.1f}',
                        ha='center', va='bottom')
            
        plt.tight_layout()
        plt.savefig('average_bandwidth_comparison.png')
        plt.close()
        network_manager.historical_data.to_csv("network_data_with_dynamic_hosts.csv", index=False)
        print("\nDataset saved to 'network_data_with_dynamic_hosts.csv'")
        
        # Cleanup
        network_manager.stop_network()
        print("\nNetwork stopped and cleaned up.")

if _name_ == "_main_":
    main()