import numpy as np

def _get_max_speed_road(dict_edge):
    
    #returns the max speed in m/s
    
    try:
        if type(dict_edge['maxspeed']) is not list:
            speed = dict_edge['maxspeed'].split(" ", 1)
            if speed[0].isdigit():
                max_speed = int(speed[0])

                try:
                    if speed[1] == 'mph':
                        #print('mph')
                        max_speed = max_speed/2.237
                    else:
                        if speed[1] == 'knots':
                            max_speed = max_speed/1.944

                except IndexError:
                    #kph
                    max_speed = max_speed/3.6

                return max_speed
            else:
                return np.nan
        else:
            max_speed_avg = 0
            for speed in dict_edge['maxspeed']:
                speed = speed.split(" ", 1)
                if speed[0].isdigit():

                    max_speed = int(speed[0])
                
                    try:
                        if speed[1] == 'mph':
                            max_speed = max_speed/2.237
                        else:
                            if speed[1] == 'knots':
                                max_speed = max_speed/1.944

                    except IndexError:
                        #kph
                        max_speed = max_speed/3.6

                    max_speed_avg = max_speed_avg + max_speed

            max_speed_avg = int(max_speed_avg/len(dict_edge['maxspeed']))
            
            if max_speed_avg > 0:
                return max_speed_avg
            else:
                return np.nan
            
    except KeyError:
        return np.nan
        
    return np.nan

def _calc_mean_max_speed(dict_edge, max_speed_mean_overall, counter_max_speeds):
    #returns the max speed in m/s
    try:
        if type(dict_edge['maxspeed']) is not list:
            speed = dict_edge['maxspeed'].split(" ", 1)
            if speed[0].isdigit():
                max_speed = int(speed[0])

                try:
                    if speed[1] == 'mph':
                        #print('mph')
                        max_speed = max_speed/2.237
                    else:
                        if speed[1] == 'knots':
                            max_speed = max_speed/1.944

                except IndexError:
                    #kph
                    max_speed = max_speed/3.6

                max_speed_mean_overall = max_speed_mean_overall + max_speed
                counter_max_speeds = counter_max_speeds + 1
            
        else:
            
            for speed in dict_edge['maxspeed']:
                speed = speed.split(" ", 1)
                if speed[0].isdigit():
                    max_speed = int(speed[0])
                    
                    try:
                        if speed[1] == 'mph':
                            #print('mph')
                            max_speed = max_speed/2.237
                        else:
                            if speed[1] == 'knots':
                                max_speed = max_speed/1.944

                    except IndexError:
                        #kph
                        max_speed = max_speed/3.6

                    max_speed_mean_overall = max_speed_mean_overall + max_speed
                    counter_max_speeds = counter_max_speeds + 1

            #max_speed_avg = int(max_speed_avg/len(dict_edge['maxspeed']))
            
            ##if max_speed_avg > 0:
            #    return max_speed_avg
            #else:
            #    return np.nan
            
    except KeyError:
        pass
        
    return max_speed_mean_overall, counter_max_speeds
    #return np.nan
