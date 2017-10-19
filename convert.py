def format_data(xfile, sensorNum, sensorAxis, sheetNum):
    for i in range(data.shape[0]/sensorNum) : # 
    newRow = np.empty(0) 
    for j in range(sensorNum):
        newSensorValues = data.iloc[3*i+j, 1:7].as_matrix()  # make every 3 rows into one
        newRow = np.append(newRow,newSensorValues)

    newSet = np.vstack((newSet, newRow))
    
newSet