     for i in range(len(transport_routes)):
        
        print(transport_routes[i]['nodes'])
        
        nodes_route = transport_routes[i]['nodes']
        
        filtered_nodes_osm = []
        
        fig, ax = ox.plot_graph(inst.network.G_walk, show=False, close=False)
        
        for u in route['nodes']:
            nodeu = api_osm.NodeGet(u)
            node_point = (nodeu['lat'], nodeu['lon'])
            
            ax.scatter(nodeu['lon'], nodeu['lat'], c='blue')
            
            print(node_point)
            nn = ox.get_nearest_node(G_drive, node_point)
            if nn not in filtered_nodes_osm:
                filtered_nodes_osm.append(nn)

        print('filtered nodes')
        print(filtered_nodes_osm)

        plt.show()
        if len(filtered_nodes_osm) > 1:
            #nc = ['r' if (node in filtered_nodes_osm) else '#336699' for node in G_drive.nodes()]
            #ns = [12 if (node in filtered_nodes_osm) else 6 for node in G_drive.nodes()]
            #fig, ax = ox.plot_graph(G_drive, node_size=ns, node_color=nc, node_zorder=2, save=True, filename='route_nodes_antwerp')
        
            #fig, ax = ox.plot_graph_route(G_drive, filtered_nodes_osm, route_linewidth=6, node_size=0, bgcolor='k')
        break

 if poi['geometry'].geom_type == 'Polygon':
    pass
    '''
    try:
        G_neigh = ox.graph_from_polygon(poi['geometry'], network_type='walk', retain_all=True)
        print('size neigh', len(G_neigh))

        n = {
            'index': index,
            'name': zone_name,
            'polygon': poi['geometry']
        }

        zones.append(n)
    except TypeError:
        pass
    '''


 tags = {
        'place':'suburb',
        #'place':'quarter',
    }
    #'place'='quarter'
    
    poi_zones = ox.pois_from_polygon(polygon_drive, tags=tags)
    
    print('number of zones: ', len(poi_zones))

    zones = []

    for index, poi in poi_zones.iterrows():
        if str(poi['name']) != 'nan':
            zone_name = str(poi['name'])
            print(zone_name)
        
            #print(poi['geometry'].geom_type)

            if poi['geometry'].geom_type == 'Polygon':
                pass
                '''
                try:
                    G_neigh = ox.graph_from_polygon(poi['geometry'], network_type='walk', retain_all=True)
                    print('size neigh', len(G_neigh))

                    n = {
                        'index': index,
                        'name': zone_name,
                        'polygon': poi['geometry']
                    }

                    zones.append(n)
                except TypeError:
                    pass
                '''

            #search for the AREA of the zone

            if poi['geometry'].geom_type == 'Point':

                distance = 500 
                zone_center_point = (poi.geometry.centroid.y, poi.geometry.centroid.x)
                
                #distance???
                G_neigh = ox.graph_from_point(zone_center_point, network_type='walk', retain_all=True)
                #print('size neigh', len(G_neigh))

                north, south, east, west = ox.bbox_from_point(zone_center_point, distance)
                polygon = Polygon([(west, south), (east, south), (east, north), (west, north)])

                n = {
                    'index': index,
                    'name': zone_name,
                    'polygon': polygon,
                    'center_point_y': poi.geometry.centroid.y,
                    'center_point_x': poi.geometry.centroid.x,
                    'center_distance': distance
                }

                zones.append(n)   



'''
    #retrieve bus stops
    tags = {
        'highway':'bus_stop',
    }
    poi_bus_stops = ox.pois_from_polygon(polygon_drive, tags=tags)
    '''

'''
    bus_stops = []
    for index, poi in poi_bus_stops.iterrows():
        if poi['highway'] == 'bus_stop':
            bus_stop_point = (poi.geometry.centroid.y, poi.geometry.centroid.x)
            
            geom, u, v = ox.get_nearest_edge(G_walk, bus_stop_point)
            bus_stop_node_walk = min((u, v), key=lambda n: ox.great_circle_vec(poi.geometry.centroid.y, poi.geometry.centroid.x, G_walk.nodes[n]['y'], G_walk.nodes[n]['x']))
            
            geom, u, v = ox.get_nearest_edge(G_drive, bus_stop_point)
            bus_stop_node_drive = min((u, v), key=lambda n: ox.great_circle_vec(poi.geometry.centroid.y, poi.geometry.centroid.x, G_drive.nodes[n]['y'], G_drive.nodes[n]['x']))
            
            d = {
                'stop_id': index,
                'osmid_walk': bus_stop_node_walk,
                'osmid_drive': bus_stop_node_drive,
                'lat': poi.geometry.centroid.y,
                'lon': poi.geometry.centroid.x,
                'itid': -1
            }
        
            bus_stops.append(d)

    bus_stops = pd.DataFrame(bus_stops)
    bus_stops.set_index(['stop_id'], inplace=True)
    '''

    #also adds more information on the arcs like -> situation of the road => traffic (slower), accident etc
#function with parameter: time and returns the current speed of the road
class Arc:

    def __init__(self, origin, destination):
        self.origin = origin
        self.destination = destination
        self.connection_type = "indirect"
        self.distance_vehicle = np.iinfo(np.int64).max
        self.distance_walking = np.iinfo(np.int64).max
        self.fastest_time = np.iinfo(np.int64).max
        self.fastest_path = []
        self.traffic_flow = []
        self.max_flow = 0
        #self.paths -> store all possible paths between two nodes?
        self.max_speed = -1
        # normal : average speed
        # traffic : 1/2 average speed ?
        # accident : 1/4 average speed ?
        
    def set_vehicle_distance(self, distance):
        self.distance_vehicle = distance
        self.connection_type = "direct"

    #add current time as a parameter maybe?
    def get_expected_travel_time(self, travel_mode):

        if self.connection_type == "direct":
            speed = travel_mode.speed
            if travel_mode.mean_transportation == "bus":
                distance = self.distance_vehicle
            expected_travel_time = int(math.ceil(distance/speed))
        else:
            expected_travel_time = np.iinfo(np.int64).max
        
        return expected_travel_time

    def set_walking_distance(self, distance):
        self.distance_walking = distance

    #also give different speeds such as bus speed, car speed, bike speed whatever
    def set_max_speed(self, max_speed):
        self.max_speed = max_speed
    
    #traffic        
    def add_traffic_flow(self, mean, std):
        self.traffic_flow.append({"mean": mean, "std": std})

    def add_accident(self, time):
        accident = True

        if accident is True:
            std = 600
            peak = 600 
            self.add_traffic_flow(time+peak, std)

    #traffic frequency at current time "time" is given by all pdfs in the arc
    def get_traffic_flow(self, time):
        mean = 1600 
        std = 800
        traffic_flow = norm.pdf(x=time, loc=mean, scale=std)*1000

    def return_current_speed(self, travel_mode, time):    
        current_speed = int(math.ceil(3*self.max_speed/4))

        return current_speed

class Node:

    def __init__(self, index, coord):
        self.index = index
        self.coord = coord
        #self.adj = []
        self.incoming_arcs = []
        self.outgoing_arcs = []

class Coordinate:

    def __init__(self, x, y):
        self.x = x
        self.y = y

class Shape:

    def __init__(self, shape_type):
        self.shape_type = shape_type

    def square(self, side_length):
        self.side_length = side_length

class Neighborhood:

    def __init__(self, name, coord, shape):
        self.name = name
        self.coord = coord
        self.shape = shape
        #self.radius = radius
        #self.area_type = area_type
        #self.origin_demand = origin_demand
        #self.destination_demand = destination_demand

    #returns a random coordinate in the given neighborhood
    def get_random_coord(self):
        if (self.shape.shape_type == "square"):
            x = randint(self.coord.x, self.coord.x + self.shape.side_length + 1) 
            y = randint(self.coord.y, self.coord.y + self.shape.side_length + 1)
        r_coord = Coordinate(x, y)
        return r_coord

    if sys.argv[i] == "--city_max":
            city_max = Coordinate(int(sys.argv[i+1]), int(sys.argv[i+2]))

            if sys.argv[i+3] == "km":
                city_max.x = city_max.x*1000
                city_max.y = city_max.y*1000

    if sys.argv[i] == "--stop_area_spacing":
            stop_area_spacing = Coordinate(int(sys.argv[i+1]), int(sys.argv[i+2]))

            if sys.argv[i+3] == "km":
                stop_area_spacing.x = stop_area_spacing.x*1000
                stop_area_spacing.y = stop_area_spacing.y*1000

    if sys.argv[i] == "--neighborhoods":
            num_neighborhoods = int(sys.argv[i+1])
            neighborhoods = []
            
            k=i+2

            name = sys.argv[k]
            k = k+1
            x = int(sys.argv[k])
            if x == -1:
                x = int(city_max.x/2)
            k = k+1
            y = int(sys.argv[k])
            if y == -1:
                y = int(city_max.y/2)
            coord = Coordinate(x, y)
            k = k+1
            shape = Shape(sys.argv[k])

            if shape.shape_type == "square":
                k = k+1
                side_length = int(sys.argv[k])
                if side_length == -1:
                    side_length = city_max.x/4
                shape.square(side_length)

            neighborhood = Neighborhood(name, coord, shape)
            neighborhoods.append(neighborhood)
            
            k = k+1
            mean = int(sys.argv[k])
            k = k+1
            std = int(sys.argv[k])
            k = k+1
            num_requests = int(sys.argv[k])
            #print(mean, std, num_requests)
            origin_demand = RequestDistribution(mean, std, num_requests)
            origin_demand.set_origin_neighborhood(neighborhood)
            request_demand.append(origin_demand)

            k = k+1
            mean = int(sys.argv[k])
            k = k+1
            std = int(sys.argv[k])
            k = k+1
            num_requests = int(sys.argv[k])
            #print(mean, std, num_requests)
            destination_demand = RequestDistribution(mean, std, num_requests)
            destination_demand.set_destination_neighborhood(neighborhood)
            request_demand.append(destination_demand)
 
            #print("neighborhoods size", len(neighborhoods))


    class TravelMode:

    def __init__(self, mean_transportation, speed):
        self.mean_transportation = mean_transportation
        self.speed = speed

    def calculate_avg_speed_route(route, G_drive, avg_uber_speed_data):
    #better to change to: calculate travel time route ? based on speed info, calculate the travel time

    alpha_speed = 0.5
    default_speed = 30
    
    prev_node = -1
    avg_speed_route = 0
    num_edges = 0
    dict_edge = {}
    
    #do a 24 hour loop for hour
    hour = 0
    for curr_node in route:
        if prev_node != -1:
            
            #first checks if there is speed information available from uber speed data 
            #second checks if there is max speed information available from OSM
            #otherwise considers default_speed
            try:
                avg_speed_route = avg_speed_route + avg_uber_speed_data.loc[(prev_node,curr_node,hour), 'speed_mph_mean']
                num_edges = num_edges + 1
            except KeyError:
                dict_edge = G_drive.get_edge_data(prev_node, curr_node)
                dict_edge = dict_edge[0]
                
                max_speed = get_max_speed_road(dict_edge)

                if math.isnan(max_speed):
                    pass
                else:
                    avg_speed_route = avg_speed_route + max_speed
                
                num_edges = num_edges + 1
                
        prev_node = curr_node

    if num_edges > 0:
        avg_speed_route = int(avg_speed_route/num_edges)
    else:
        avg_speed_route = default_speed

    return avg_speed_route

    '''
    for hour in range(24):
        for index_o, origin_stop in bus_stops.iterrows():
            for index_d, destination_stop in bus_stops.iterrows():
                travel_time_row = {}
                travel_time_row['stop_origin_id'] = index_o
                travel_time_row['stop_destination_id'] = index_d
                travel_time_row['hour'] = hour

                if  origin_stop['osmid_drive'] != destination_stop['osmid_drive']:
                    try:
                        path_length = dict_shortest_path_length[origin_stop['osmid_drive']][destination_stop['osmid_drive']]
                        path = dict_shortest_path[origin_stop['osmid_drive']][destination_stop['osmid_drive']]
                        
                        #converter velocidades para m/s
                        #aqui fazer a separação (if inst.speed_info = 'uber_speed')
                        #(if inst.speed_info = 'max_speed')
                        #max speed
                        #uber speed
                        #if speed_data != 'max':
                            #travel_time_path = calculate_travel_time_path_uber_speed(G_drive, path, path_length, dict_shortest_path_length, avg_uber_speed_data)

                        travel_time_row['travel_time'] = path_length
                    except KeyError:
                        travel_time_row['travel_time'] = -1
                    
                else:
                    travel_time_row['travel_time'] = 0

                travel_time_matrix.append(travel_time_row)

    travel_time_matrix = pd.DataFrame(travel_time_matrix)
    travel_time_matrix.set_index(['stop_origin_id', 'stop_destination_id', 'hour'], inplace=True)
    '''


    '''
    print('Now genarating distances_data and max_speed_data')
    
    distance_matrix = []
    avg_curr_speed_matrix = []

    for index_o, origin_stop in bus_stops.iterrows():
        row_origin = {}
        row_speed = {}
        row_origin['stop_origin_id'] = index_o
        row_speed['stop_origin_id'] = index_o
        for index_d, destination_stop in bus_stops.iterrows():
            if  origin_stop['osmid_drive'] != destination_stop['osmid_drive']:
                try:
                    #route is a list with all nodes in the route
                    route = nx.shortest_path(G_drive, origin_stop['osmid_drive'], destination_stop['osmid_drive'], weight='length')
                    
                    #use each node in route to calculate travel time
                    route_length = nx.shortest_path_length(G_drive, origin_stop['osmid_drive'], destination_stop['osmid_drive'], weight='length')

                    #calculation speed information   
                    avg_speed_route = calculate_avg_speed_route(route, G_drive, avg_uber_speed_data)
                    row_speed[index_d] = avg_speed_route
                    
                    #updating row of dataframe by adding a column
                    row_origin[index_d] = route_length
                    
                except nx.NetworkXNoPath:
                    #there is no drivable path between stop1 and stop2
                    #updating row of dataframe by adding a column
                    route_length = -1
                    row_origin[index_d] = route_length
                    row_speed[index_d] = -1               
            else:
                #updating row of dataframe by adding a column
                route_length = 0
                row_origin[index_d] = route_length
                row_speed[index_d] = 0 

        distance_matrix.append(row_origin)
        avg_curr_speed_matrix.append(row_speed)

    distance_matrix = pd.DataFrame(distance_matrix)
    distance_matrix.set_index(['stop_origin_id'], inplace=True)

    #add hour as an index later on
    avg_curr_speed_matrix = pd.DataFrame(avg_curr_speed_matrix)
    avg_curr_speed_matrix.set_index(['stop_origin_id'], inplace=True)
    
    print(avg_curr_speed_matrix.head())
    
    useless_bus_stop = True
    while useless_bus_stop:
        useless_bus_stop = False
        for index1, stop1 in bus_stops.iterrows():
            unreachable_nodes = 0
            for index2, stop2 in bus_stops.iterrows():

                if distance_matrix.loc[index1, index2] == -1:
                    unreachable_nodes = unreachable_nodes + 1
            if unreachable_nodes == len(bus_stops) - 1:
                bus_stops = bus_stops.drop(index1)
                distance_matrix = distance_matrix.drop(index1, axis=0) #drop the row
                distance_matrix = distance_matrix.drop(index1, axis=1) #drop the column
                avg_curr_speed_matrix = avg_curr_speed_matrix.drop(index1, axis=0) #drop the row
                avg_curr_speed_matrix = avg_curr_speed_matrix.drop(index1, axis=1) #drop the column
                useless_bus_stop = True
    
    #print(avg_curr_speed_matrix.max(), avg_curr_speed_matrix.min())
       
    #fazer comparação com a distancia em coordenadas (euclideana)?
    print('number of bus stops after removal: ', len(bus_stops))
    #acho que aqui fim do antigo calculco de distancia + velocidade
    '''



    '''
    G_wstops = G_drive
    bus_stops_walking_threshold = []  
    for index, poi in bus_stops.iterrows():
        if poi['highway'] == 'bus_stop':
            G_wstops.add_node(poi['osmid'])
            G_wstops.nodes[poi['osmid']]['y'] = poi.geometry.centroid.y
            G_wstops.nodes[poi['osmid']]['x'] = poi.geometry.centroid.x
            G_wstops.nodes[poi['osmid']]['osmid'] = poi['osmid']
            G_wstops.nodes[poi['osmid']]['highway'] = poi['highway']
            bus_stops_walking_threshold.append(index)
    '''    
    
    '''
    for (u,v,k) in G_drive.edges(data=True):
        #print(G.edges[u,v])
        print (u,v,k)
    '''

    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
    #X = scaler.fit_transform(X)


    #normalizing the data
    #the validation set approach
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=6)

    '''    
    def add_graphs(self, G_drive, polygon_drive, G_walk, polygon_walk):
        self.G_drive = G_drive
        self.polygon_drive = polygon_drive
        self.G_walk = G_walk
        self.polygon_walk = polygon_walk

    def add_bus_stops(self, bus_stop_nodes):
        self.bus_stop_nodes = bus_stop_nodes
        self.num_stations = len(bus_stop_nodes)
    

    def add_distance_matrix(self, distance_matrix):
        self.distance_matrix = distance_matrix
    
    def fastest_node(self, fastest_time, finalized):
        fastest_time_v = np.iinfo(np.int64).max
        fastest_index_v = -1

        for v in range(len(self.nodes)):
            if fastest_time[v] < fastest_time_v and finalized[v] == False:
                fastest_time_v = fastest_time[v]
                fastest_index_v = v

        return fastest_index_v

    def dijkstras(self, source, travel_mode):
        fastest_time = [np.iinfo(np.int64).max] * len(self.nodes)
        fastest_time[source] = 0
        finalized = [False] * len(self.nodes)

        for count in range(len(self.nodes)):

            src = self.fastest_node(fastest_time, finalized) 
            finalized[src] = True

            for dest in range(len(self.nodes)):
                expected_travel_time_src_dest = self.arcs[src][dest].get_expected_travel_time(travel_mode)
                if finalized[dest] == False and fastest_time[dest] > fastest_time[src] + expected_travel_time_src_dest:
                    fastest_time[dest] = fastest_time[src] + expected_travel_time_src_dest

        return fastest_time

    def update_fastest_time_bus(self):
        for source in range(len(self.nodes)):
            t = self.get_travel_mode_index("bus")
            fastest_time = self.dijkstras(source, self.travel_modes[t])
            for dest in range(len(self.nodes)):
                self.arcs[source][dest].fastest_time = fastest_time[dest]

    def calculate_distance_euclidean(self, node1, node2):
        return int(math.ceil(((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2) ** .5))  # was vroeger int, maar gaf afrondingsproblemen

    def calculate_distance(self, node1, node2):
        return int(math.ceil(((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2) ** .5))  # was vroeger int, maar gaf afrondingsproblemen
        
    def update_network(self, nodes):
        self.nodes = nodes
        #generate the arcs in the network
        self.generate_vehicle_arcs()
        #time dist matrix
        self.update_fastest_time_bus()

    def generate_vehicle_arcs(self):
        self.arcs = np.ndarray((len(self.nodes), len(self.nodes)), dtype=object)
        radius_direct_road = 2000
        #add the probability later
        for i in range(len(self.nodes)):
            max_value = np.iinfo(np.int64).max
            cn_id = []
            for j in range(len(self.nodes)):
                if i != j:
                    distance = self.calculate_distance_euclidean(self.nodes[i].coord, self.nodes[j].coord)
                    if (distance <= radius_direct_road):
                        cn_id.append(j) 

            #every i should leave this loop with at least one outgoing arc

            for k in range(len(cn_id)):
                adj = cn_id[k]
                distance = self.calculate_distance(self.nodes[i].coord, self.nodes[adj].coord)
                arc = Arc(self.nodes[i].index, self.nodes[adj].index)
                arc.set_vehicle_distance(distance)
                self.arcs[self.nodes[i].index][self.nodes[adj].index] = arc

        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if self.arcs[self.nodes[i].index][self.nodes[j].index] is None:
                    self.arcs[self.nodes[i].index][self.nodes[j].index] = Arc(self.nodes[i].index, self.nodes[j].index)
                    #distance is infinite. dijkstra's algorithm calculate fastest time between nodes that are not directly connected
    
                    
    def return_estimated_arrival_bus(self, origin, destination, stops_origin, stops_destination):
        max_eta_bus = -1

        for i in range(len(stops_origin)):
            for j in range(len(stops_destination)):
                if self.arcs[stops_origin[i]][stops_destination[j]].fastest_time > max_eta_bus:
                    max_eta_bus = self.arcs[stops_origin[i]][stops_destination[j]].fastest_time

        return max_eta_bus

    def return_estimated_arrival_walk(self, origin_coord, destination_coord):

        min_distance = self.calculate_distance_euclidean(origin_coord, destination_coord)
        horizon_coord = Coordinate(destination_coord.x, origin_coord.y)
        #vertical_coord = Coordinate(origin_coord.x, destination_coord.y)
        #test if max distance is the same for both vertical/horizontal coord
        max_distance = self.calculate_distance_euclidean(origin_coord, horizon_coord) + self.calculate_distance_euclidean(horizon_coord, destination_coord)
        i = self.get_travel_mode_index("walking")
        avg_distance = int(math.ceil((max_distance+min_distance)/2))
        speed = self.travel_modes[i].speed
        eta_walk = int(math.ceil(avg_distance/speed))

        return eta_walk

    def get_travel_time_matrix(self, travel_mode_string, interval):
        #k = self.get_travel_mode_index(travel_mode_string)
        #travel_time_k = [[None for j in range(len(self.nodes))] for i in range(len(self.nodes))]
        travel_time = np.ndarray((len(self.nodes), len(self.nodes)), dtype=np.int64)
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if travel_mode_string == "bus":
                    travel_time[i][j] = self.arcs[i][j].fastest_time

        return travel_time

    def update_neighborhoods(self):
        for i in range(self.num_city_center):
            #choose the coordinates for the city center
            cx = 1
            cy = 1 
            radius = 20 
            area_type = 1 #types of area, which could mean, center, high density populated area, low density populated area
            city_center = CityArea(cx, cy, radius, area_type)
            self.city_centers.append(city_center) 
    
    '''

    '''
    rmse_val = []
    best_k = -1
    min_rmse = 999999
    #max_score = -1
    for k in range(40):
        k = k+1
        #train the model
        regressor = KNeighborsRegressor(n_neighbors=k)
        modelknn = regressor.fit(X_train, y_train)
        #prediction
        y_pred = regressor.predict(X_test)
        error = sqrt(mean_squared_error(y_test,y_pred)) #calculate rmse
        #scoreknn = modelknn.score(X_test, y_test) #calculate accuracy 
        rmse_val.append(error)
        if error < min_rmse:
            min_rmse = error
            best_k = k
            #max_score = scoreknn
    print("best k, min rmse:", best_k, min_rmse)
    '''

                    '''
                if inst.request_demand[r].origin_neighborhood is None:
                    originx = randint(0, inst.network.city_max.x + 1) #that can create a coordinate 
                    originy = randint(0, inst.network.city_max.y + 1)
                    origin = Coordinate(originx, originy)
                else:
                    origin = inst.request_demand[r].origin_neighborhood.get_random_coord()

                if inst.request_demand[r].destination_neighborhood is None:
                    destinationx = randint(0, inst.network.city_max.x + 1)
                    destinationy = randint(0, inst.network.city_max.y + 1)
                    destination = Coordinate(destinationx, destinationy)
                else:
                    #print(inst.num_requests)
                    destination = inst.request_demand[r].destination_neighborhood.get_random_coord()
                '''
                
                '''
                if inst.network.return_estimated_arrival_walk(origin, destination) > inst.max_walking: #if distance between origin and destination is too small the person just walks
                    #change j to station_id or something similar. node_id
                    #calculates the stations which are close enough to the origin and destination of the request
                    
                    for j in range(inst.network.num_stations):
                        #change the append for the "index"?
                        
                        if inst.network.return_estimated_arrival_walk(origin, stations[j].coord) <= inst.max_walking:
                            stops_origin.append(j)

                        if inst.network.return_estimated_arrival_walk(destination, stations[j].coord) <= inst.max_walking:
                            stops_destination.append(j)
                '''


    '''
    for index1, stop1 in bus_stops.iterrows():
        lat1 = stop1.geometry.centroid.y
        lng1 = stop1.geometry.centroid.x
        stop1_location = (lat1, lng1)
        stop1_node = ox.get_nearest_node(G_drive, stop1_location)
        for index2, stop2 in bus_stops.iterrows():
            if (stop1['osmid'] != stop2['osmid']):
                lat2 = stop2.geometry.centroid.y
                lng2 = stop2.geometry.centroid.x
                stop2_location = (lat2, lng2)
                
                #geom2, u2, v2 = ox.get_nearest_edge(G_drive, (lat2, lng2))
                #nn = min((u2, v2), key=lambda n: ox.great_circle_vec(lat2, lng2, G_drive.nodes[n]['y'], G_drive.nodes[n]['x']))
                #print(nn)
                
                stop2_node = ox.get_nearest_node(G_drive, stop2_location)
                #print(stop2_node)
                #route = nx.shortest_path(G, origin_node, destination_node, weight='length')
                try:
                    route_length = nx.shortest_path_length(G_drive, stop1_node, stop2_node, weight='length')
                    #print('shortest path (meters): ', route_length)
                    #print(route_length)
                    #if route_length > 0?
                    count_calc = count_calc + 1
                    #if count_calc % 1000 == 0:
                        #print(count_calc)
                    d = {
                        'origin_osmid': stop1['osmid'],
                        'destination_osmid': stop2['osmid'],
                        'shortest_path_length': route_length
                    }
                    distance_matrix.append(d)
                except nx.NetworkXNoPath:
                    #print('achou')
                    pass
    '''                

    

    #print(distance_matrix.index)

    '''
    for index1, stop1 in bus_stops.iterrows():
        for index2, stop2 in bus_stops.iterrows():
            if (stop1['osmid'] != stop2['osmid']):
                if distance_matrix.index.isin([(stop1['osmid'], stop2['osmid'])]).any():
                    print(distance_matrix.loc[(stop1['osmid'], stop2['osmid']), 'shortest_path_length'])
    '''                
    
    
    '''            
    origin_point = (51.22939215, 4.41408171)
    destination_point = (51.222578, 4.410082)

    origin_node = ox.get_nearest_node(G, origin_point)
    destination_node = ox.get_nearest_node(G, destination_point)
    #print(origin_node, destination_node)
    route = nx.shortest_path(G, origin_node, destination_node, weight='length')
    route_length = nx.shortest_path_length(G, origin_node, destination_node, weight='length')
    #fig, ax = ox.plot_graph_route(G, route, origin_point=origin_point, destination_point=destination_point, file_format='svg', filename='shortest_path')
    print('shortest path (meters): ', route_length)
    '''

    

    #count_stops = 0


    #for this is necessary to search for the nearest node in the network + distance da location da bus_stop pro node
    #station_point = for each stop in gdf
    #given a location, return the stops arround
    #nearest_node_station = ox.get_nearest_node(G, station_point) + distance to station

    #all_simple_paths or shortest_simple_paths to generate all paths -> teria q calcular a distancia de cada um depois de retornar o caminho
    #stats = ox.basic_stats(G)
    #G = ox.add_edge_lengths(G)

    #G_projected = ox.project_graph(G)
    #fig, ax = ox.plot_graph(G_projected, save=True, file_format='svg', filename='after_stops')
    #X = {4.41408171} lon
    #Y = {51.22939215} lat
    #G2 = ox.graph_from_point(location_point, distance=500, distance_type='network', network_type='walk')
    #ox.get_nearest_nodes(G_projected, )
    #for index, poi in gdf.iterrows():
        #if poi['element_type'] = '':
        #print(poi['access'])
    #for j in range(len(nd_samples)):
        #print(len(nd_samples[j].demand))
        #for k in range(len(nd_samples[j].demand)):
            #print(nd_samples[j].demand[k])

for stop1 in bus_stop_nodes:
        lat1 = G_walk.nodes[stop1]['y']
        lng1 = G_walk.nodes[stop1]['x']
        stop1_location = (lat1, lng1)
        stop1_node = ox.get_nearest_node(G_drive, stop1_location)
        for stop2 in bus_stop_nodes:
            if (stop1 != stop2):
                lat2 = G_walk.nodes[stop2]['y']
                lng2 = G_walk.nodes[stop2]['x']
                stop2_location = (lat2, lng2)
                stop2_node = ox.get_nearest_node(G_drive, stop2_location)
                
                #geom2, u2, v2 = ox.get_nearest_edge(G_drive, (lat2, lng2))
                #nn = min((u2, v2), key=lambda n: ox.great_circle_vec(lat2, lng2, G_drive.nodes[n]['y'], G_drive.nodes[n]['x']))
                #print(nn)
                #print(stop2_node)
                
                try:
                    route = nx.shortest_path(G_drive, stop1_node, stop2_node, weight='length')
                    route_length = nx.shortest_path_length(G_drive, stop1_node, stop2_node, weight='length')

                    #print("origin node: ", stop1_node)
                    prev_node = -1
                    length_route = 0
                    avg_speed_route = 0
                    num_edges = 0
                    dict_edge = {}
                    unkown_speed = 0
                    for curr_node in route:
                        if prev_node != -1:
                            dict_edge = G_drive.get_edge_data(prev_node, curr_node)
                            dict_edge = dict_edge[0]
                            #num_edges = num_edges + 1
                            '''
                            try:
                                length_route = length_route + dict_edge['length']
                            except KeyError:
                                pass # 'unkown length' 
                            '''
                            try:
                                if type(dict_edge['maxspeed']) is not list:
                                    if dict_edge['maxspeed'].isdigit():
                                        avg_speed_route = avg_speed_route + int(dict_edge['maxspeed'])
                                        num_edges = num_edges + 1
                                else:
                                    maxspeedavg = 0
                                    for speed in dict_edge['maxspeed']:
                                        if speed.isdigit():
                                            maxspeedavg = maxspeedavg + int(speed)
                                    maxspeedavg = int(maxspeedavg/len(dict_edge['maxspeed']))
                                    avg_speed_route = avg_speed_route + maxspeedavg
                                    num_edges = num_edges + 1
                            except KeyError:
                                #unkown_speed = unkown_speed + 1 # 'unkown speed' 
                                avg_speed_route = avg_speed_route + default_max_speed
                                num_edges = num_edges + 1
                            '''
                            try:
                                print(dict_edge['maxspeed:variable'])
                            except KeyError:
                                #print('no variable speed tag')
                                pass
                            '''
                        prev_node = curr_node

                    #print("destination node: ", stop2_node)    
                    
                    #if unkown_speed > 0:
                        #print('num_edges', num_edges)
                        #print('unkown speeds: ', unkown_speed)
                    
                    if stop1_node != stop2_node:
                        if num_edges > 0:
                            avg_speed_route = int(avg_speed_route/num_edges)
                        else:
                            avg_speed_route = default_max_speed
                    else:
                        print(route_length)
                        fig, ax = ox.plot_graph_route(G_drive, route, origin_point=stop1_location, destination_point=stop2_location)
                        #num edges equals 0
                        #else:
                        

                        #print("route size: ", len(route))
                        #for node in route:
                        #    print(node)
                        #print('origin', stop1_node)
                        #print('destination', stop2_node)
                        


                    #print(avg_speed_route)

                    d = {
                        'origin_osmid': stop1,
                        'origin_osmid_drive': stop1_node,
                        #'origin_id': G_walk.nodes[stop1]['itid'],
                        'origin_id': -1,
                        'destination_osmid': stop2,
                        'destination_osmid_drive': stop2_node,
                        #'destination_id': G_walk.nodes[stop2]['itid'],
                        'destination_id': -1,
                        'shortest_path_length': route_length
                    }
                    distance_matrix.append(d)
                    count_calc = count_calc + 1
                except nx.NetworkXNoPath:
                    #there is no drivable path between stop1 and stop2
                    route_length = -1
                    d = {
                        'origin_osmid': stop1,
                        'origin_osmid_drive': stop1_node,
                        #'origin_id': G_walk.nodes[stop1]['itid'],
                        'origin_id': -1,
                        'destination_osmid': stop2,
                        'destination_osmid_drive': stop2_node,
                        #'destination_id': G_walk.nodes[stop2]['itid'],
                        'destination_id': -1,
                        'shortest_path_length': route_length
                    }
                    distance_matrix.append(d)
                
                    
                           
            else:
                route_length = 0
                d = {
                        'origin_osmid': stop1,
                        'origin_osmid_drive': stop1_node,
                        #'origin_id': G_walk.nodes[stop1]['itid'],
                        'origin_id': -1,
                        'destination_osmid': stop1,
                        'destination_osmid_drive': stop1_node,
                        #'destination_id': G_walk.nodes[stop2]['itid'],
                        'destination_id': -1,
                        'shortest_path_length': route_length
                    }
                distance_matrix.append(d)

'''
    def fastest_node2(self, fastest_time, finalized, leng):
        fastest_time_v = np.iinfo(np.int64).max
        fastest_index_v = -1

        for v in range(leng):
            if fastest_time[v] < fastest_time_v and finalized[v] == False:
                fastest_time_v = fastest_time[v]
                fastest_index_v = v

        return fastest_index_v

    def dijkstras2(self, source, graph, leng):
        fastest_time = [np.iinfo(np.int64).max] * leng
        fastest_time[source] = 0
        finalized = [False] * leng

        for count in range(leng):

            src = self.fastest_node2(fastest_time, finalized, leng) 
            finalized[src] = True

            for dest in range(leng):
                #expected_travel_time_src_dest = self.arcs[src][dest].get_expected_travel_time(travel_mode)
                if graph[src][dest] > 0 and finalized[dest] == False and fastest_time[dest] > fastest_time[src] + graph[src][dest]:
                    fastest_time[dest] = fastest_time[src] + graph[src][dest]

        return fastest_time
    '''
 '''
        nodesg = 9
        graph = [[0 for column in range(nodesg)]  
                    for row in range(nodesg)]
        graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0], 
        [4, 0, 8, 0, 0, 0, 0, 11, 0], 
        [0, 8, 0, 7, 0, 4, 0, 0, 2], 
        [0, 0, 7, 0, 9, 14, 0, 0, 0], 
        [0, 0, 0, 9, 0, 10, 0, 0, 0], 
        [0, 0, 4, 14, 10, 0, 2, 0, 0], 
        [0, 0, 0, 0, 0, 2, 0, 1, 6], 
        [8, 11, 0, 0, 0, 0, 1, 0, 7], 
        [0, 0, 2, 0, 0, 0, 6, 7, 0] 
        ];
        nodese = 4
        edges =  [[0, 3, 4, 0],
          [0, 0, 0.5, 0],
          [0, 0, 0, 1],
          [0, 0, 0, 0]]
        test_dijks = self.dijkstras2(0, edges, nodese)
        for v in range(nodese):
            print(v, "t", test_dijks[v])
        '''
def return_estimated_arrival_walk2(self, origin_coord, destination_coord):

    distance = self.calculate_distance_euclidean(origin_coord, destination_coord)
    i = self.get_travel_mode_index("walking")
    speed = self.travel_modes[i].speed
    eta_walk = int(math.ceil(distance/speed))

    return eta_walk


def calculate_travel_time_walk(self, node1, node2, travel_mode):
    i = self.get_travel_mode_index(travel_mode)
    #now the nodes should be mapped to calculate the distance -> find the neartes station (node, bus stop, bike station whatever and then calculate)
    distance = self.calculate_distance_euclidean(node1, node2)
    speed = self.travel_modes[i].speed     
    return int(math.ceil(distance/speed))
    

def generate_vehicle_arcs2(self):
        self.arcs = np.ndarray((len(self.nodes), len(self.nodes)), dtype=object)

        #add the probability later
        for i in range(len(self.nodes)):
            max_value = np.iinfo(np.int64).max
            #stores the closest nodes: 0->left, 1->bottom, 2->right, 3->top
            #0 -> saves distance 1 -> saves node  
            closest_nodes = np.full((4,2), max_value, dtype=np.int64) 
            closest_nodes[0][1] = -1
            closest_nodes[1][1] = -1
            closest_nodes[2][1] = -1
            closest_nodes[3][1] = -1
            for j in range(len(self.nodes)):
                if i != j:
                    distance = self.calculate_distance_euclidean(self.nodes[i].coord, self.nodes[j].coord)
                    diff_x = self.nodes[i].coord.x - self.nodes[j].coord.x
                    diff_y = self.nodes[i].coord.y - self.nodes[j].coord.y
                    if abs(diff_x) < abs(diff_y):
                        if (distance < closest_nodes[0][0]) and (diff_x > 0):
                            #left
                            closest_nodes[0][0] = distance
                            closest_nodes[0][1] = j
                        else:
                            if (distance < closest_nodes[2][0]) and (diff_x < 0):
                                #right    
                                closest_nodes[2][0] = distance
                                closest_nodes[2][1] = j

                    if abs(diff_y) < abs(diff_x):
                        if (distance < closest_nodes[1][0]) and (diff_y > 0):
                            #bottom
                            closest_nodes[1][0] = distance
                            closest_nodes[1][1] = j
                        else:
                            if (distance < closest_nodes[3][0]) and (diff_y < 0):
                                #top
                                closest_nodes[3][0] = distance
                                closest_nodes[3][1] = j
            
            #distances = []
            #every i should leave this loop with at least one outgoing arc

            for k in range(4):
                if closest_nodes[k][1] != -1:
                    adj = closest_nodes[k][1]
                    distance = self.calculate_distance(self.nodes[i].coord, self.nodes[adj].coord)
                    arc = Arc(self.nodes[i].index, self.nodes[adj].index)
                    arc.set_vehicle_distance(distance)
                    self.arcs[self.nodes[i].index][self.nodes[adj].index] = arc

        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if self.arcs[self.nodes[i].index][self.nodes[j].index] is None:
                    self.arcs[self.nodes[i].index][self.nodes[j].index] = Arc(self.nodes[i].index, self.nodes[j].index)
                    #no need to set the distance. it will be used to calculate in dijkstra algoritm for those that dont have


itid = 0
for index, stop in bus_stops: 
    G_walk.nodes[stop[osmid_walk]]['itid'] = itid
    itid = itid + 1

#max_distance_stops = -1
#maxstop1 = 0
#maxstop2 = 0
'''
for index, stop1 in bus_stops:
    for index, stop2 in bus_stops:
        distance_matrix.loc[(stop1['osmid_drive'], stop2['osmid_drive']), 'origin_id'] = G_walk.nodes[stop1['osmid_walk']]['itid'] 
        distance_matrix.loc[(stop1['osmid_drive'], stop2['osmid_drive']), 'destination_id'] = G_walk.nodes[stop2['osmid_walk']]['itid']
        #if distance_matrix.loc[(stop1, stop2), 'shortest_path_length'] > max_distance_stops:
        #    max_distance_stops = distance_matrix.loc[(stop1, stop2), 'shortest_path_length']
        #    maxstop1 = stop1
        #    maxstop2 = stop2
'''
