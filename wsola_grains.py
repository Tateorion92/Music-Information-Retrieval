##split up original data into grains (aka analysis frames)

num_grains = len(data)/Ha-1
shift_range = 10

for grain in range(num_grains):
    if(grain*Ha <= (len(data)-grain_len)):
        if(grain==0):
            grains.append(data[grain*Ha:(grain*Ha)+grain_len])
        else:
            prv_g = grain-1
            des_i = prv_g*Ha+Hs     ##starting index of natural progression from previous grain
            previous = grains[prv_g]
            natural = data[des_i:des_i+grain_len]
            
            #target_region = data[grain*Ha-shift_index:grain*Ha+grain_len+shift_index+1]
            
            corr = 0
            max_corr = 0
            shift_index = 0
            #loop to find the maximally optimal succeeding grain
            for shift in range(-shift_range, shift_range+1):
                target = data[grain*Ha+shift:grain*Ha+grain_len]
                corr = np.correlate(previous, target)
                if(corr[0] > max_corr):
                    max_corr = corr[0]
                    shift_index = shift
            target = data[grain*Ha+shift_index:grain*Ha+grain_len]
            grains.append(target)

