

#endstop locations
def check_adj_nodes(dimensionsu, dimensionsv):

    for i in range(5):
        if abs(dimensionsu[i] - dimensionsv[i]) == 1:
            for k in range(5): 
                if i != k:
                    if dimensionsu[k] == dimensionsv[k]:
                        return True

    return False

def find_new_stop_location(param, network, locations, unc_locations, bus_stops):

    min_total_distance = math.inf
    loc_min_dist = -1

    #print(bus_stops)
    for loc1 in range(len(locations)):
        u = locations[loc1]['osmid_walk']
        #locations[loc1]['total_distance'] = 0
        total_distance = 0
        #print(u)
        #print(loc1)
        if loc1 not in bus_stops:
            #print('here')
            for loc2 in range(len(unc_locations)):
                v = unc_locations[loc2]['osmid_walk']
                #print('fns:', u, v)
                #print(u, v)
                if u != v:
                    #print(u, v)
                    sv = str(v)
                    if str(network.shortest_path_walk.loc[u, sv]) != 'nan':
                        total_distance += int(network.shortest_path_walk.loc[u, sv])
                    else:
                        pass
                        #print('nnnn')
                        #print(network.shortest_path_walk.loc[u, sv])
            
            #print(total_distance)
            #print(min_total_distance)
            if min_total_distance > total_distance:
                min_total_distance = total_distance
                loc_min_dist = loc1

    return loc_min_dist

def assign_location_to_nearest_stop(param, network, locations, bus_stops):

    for loc in range(len(locations)):
        u = locations[loc]['osmid_drive']
        min_dist = math.inf 

        for stop in range(len(bus_stops)):
            loc_stop = bus_stops[stop]
            
            v = locations[loc_stop]['osmid_drive']
            sv = str(v)
            try:
                dist = network.shortest_path_drive.loc[u, sv]
                if str(dist) != 'nan':
                    dist = int(dist)
                    if dist < min_dist:
                        min_dist = dist
                        locations[loc]['nearest_stop'] = stop
            except KeyError:
                pass
    
    locations_assigned_to_stop = [[] for i in range(len(bus_stops))]

    for loc in range(len(locations)):
        stop = locations[loc]['nearest_stop']
        locations_assigned_to_stop[stop].append(loc)

    return locations, locations_assigned_to_stop

def reset_location_stop(param, network, locations, locations_assigned_to_stop, bus_stops):

    #For each stop, reset its location at the location that has the minimum total distance to the other locations that assigned to the stop
    #CHECK IF THE LOCATION IS THE SAME???
    min_total_distance = math.inf
    new_stop = -1

    for loc1 in range(len(locations)):
        u = locations[loc1]['osmid_drive']
        total_distance = 0
        
        if loc1 not in bus_stops:
            for loc2 in range(len(locations_assigned_to_stop)):  
                v = locations[loc2]['osmid_drive']
                if u != v:
                    sv = str(v)
                    try:
                        if str(network.shortest_path_drive.loc[u, sv]) != 'nan':
                            total_distance += int(network.shortest_path_drive.loc[u, sv])
                    except KeyError:
                        pass
            
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                new_stop = loc1

    return new_stop

def k_medoids(param, network, locations, unc_locations, bus_stops):

    decrease_distance = True
    total_distance = math.inf

    locations, locations_assigned_to_stop = assign_location_to_nearest_stop(param, network, locations, bus_stops)

    while decrease_distance:

        for stop in range(len(bus_stops)):
            new_stop = reset_location_stop(param, network, locations, locations_assigned_to_stop[stop], bus_stops)
            bus_stops[stop] = new_stop

        locations, locations_assigned_to_stop = assign_location_to_nearest_stop(param, network, locations, bus_stops)

        #Calculate the sum of distances from all the locations to their nearest stops
        sum_distances = 0
        for loc in range(len(locations)):
            
            u = locations[loc]['osmid_drive']
            nearest_stop = locations[loc]['nearest_stop']
            loc_stop = bus_stops[nearest_stop]
            v = locations[loc_stop]['osmid_drive']
            sv = str(v)
            try:
                if str(network.shortest_path_drive.loc[u, sv]) != 'nan':
                    sum_distances += int(network.shortest_path_drive.loc[u, sv])
            except KeyError:
                pass

        if sum_distances < total_distance:
            total_distance = sum_distances
            decrease_distance = True
        else:
            decrease_distance = False

        print('k_medoids total dist:', total_distance)

    return bus_stops

def loc_is_covered(param, network, loc, locations, bus_stops):

    u = locations[loc]['osmid_walk']
    
    for stop in bus_stops:
        v = locations[stop]['osmid_walk']
        
        if u == v:
            return True

        try:
            
            sv = str(v)
            dist = network.shortest_path_walk.loc[u, sv]
            
            if str(dist) != 'nan':
                walk_time = int(math.ceil(dist/network.walk_speed))
                #print('walk time:', walk_time)
            else:
                walk_time = math.inf

            if walk_time <= param.max_walking:
                #print('covered loc:', walk_time)
                return True
        
        except KeyError:
            #print('kerror')
            pass

    return False

def update_unc_locations(param, network, locations, bus_stops):

    cov_locations = 0
    unc_locations = []

    for loc in range(len(locations)):
        
        if loc_is_covered(param, network, loc, locations, bus_stops):
            cov_locations += 1
        else:
            unc_locations.append(locations[loc])


    total_locations = len(locations)
    pct_cvr = (cov_locations/total_locations)*100

    return unc_locations, pct_cvr

def assign_stop_locations(param, network, cluster, G_clusters):

    bus_stops = []
    k = 1
    locations = []

    for u in cluster:
        
        iu = int(u)
        value = G_clusters.nodes[iu]['osmid_origin_walk']
        if not any(l.get('osmid_walk', None) == value for l in locations):
            lo = {
                'osmid_walk': G_clusters.nodes[iu]['osmid_origin_walk'],
                'osmid_drive': G_clusters.nodes[iu]['osmid_origin_drive'],
                'point': G_clusters.nodes[iu]['origin_point'],
                'total_distance': 0,
                'nearest_stop': -1,
            }
            locations.append(lo)

        value = G_clusters.nodes[iu]['osmid_destination_walk']
        if not any(l.get('osmid_walk', None) == value for l in locations):
            ld = {
                'osmid_walk': G_clusters.nodes[iu]['osmid_destination_walk'],
                'osmid_drive': G_clusters.nodes[iu]['osmid_destination_drive'],
                'point': G_clusters.nodes[iu]['destination_point'],
                'total_distance': 0,
                'nearest_stop': -1,
            }
            locations.append(ld)
                
    print('num of locations', len(locations))

    unc_locations = locations
    new_stop = find_new_stop_location(param, network, locations, unc_locations, bus_stops)
    bus_stops.append(new_stop)
    unc_locations, pct_cvr = update_unc_locations(param, network, locations, bus_stops)

    print('"%" cvr: ', pct_cvr)
    print('num of uncovered loc: ', len(unc_locations))
    print('xxx')

    num_iterations = 0

    while pct_cvr < 75 and num_iterations < 200:
        #num_iterations += 1

        k += 1
        new_stop = find_new_stop_location(param, network, locations, unc_locations, bus_stops)
        bus_stops.append(new_stop)
        
        #adjust location of stops
        bus_stops = k_medoids(param, network, locations, unc_locations, bus_stops)

        unc_locations, pct_cvr = update_unc_locations(param, network, locations, bus_stops)

        print('"%" cvr: ', pct_cvr)
        print('num of uncovered loc: ', len(unc_locations))
        #print(bus_stops)
        #print('xxx')
        num_iterations += 1

    print(bus_stops)
    stop_locations = []
    for stop in bus_stops:
        d = {
            'stop_id': locations[stop]['osmid_drive'],
            'osmid_walk': locations[stop]['osmid_walk'],
            'osmid_drive': locations[stop]['osmid_drive'],
            'lat': locations[stop]['point'].y,
            'lon': locations[stop]['point'].x,
        }
        #stop_locations_point.append(locations[stop]['point'])
        #osmid_walk_stop_locations.append(locations[stop]['osmid_walk'])
        #osmid_drive_stop_locations.append(locations[stop]['osmid_drive'])
        stop_locations.append(d)

    return stop_locations

def cluster_travel_demand(param, network, num_points=20, request_density_treshold=1):
    
    #zone_id = 0
    #for zone in network.zones:
    #distance = zone['center_distance']
    #zone_center_point = (zone['center_point_y'], zone['center_point_x'])
    #ax.scatter(zone['center_point_x'], zone['center_point_y'], c='red')
    #north, south, east, west = ox.bbox_from_point(zone_center_point, distance)
    #polygon = Polygon([(west, south), (east, south), (east, north), (west, north)])

    #SPACE PARTINIONING
    #fig, ax = ox.plot_graph(network.G_walk, show=False, close=False)
    polygon = network.polygon_walk
    minx, miny, maxx, maxy = polygon.bounds

    #ax.scatter(maxx, maxy, c='green')
    #ax.scatter(minx, miny, c='green')            
    #diffy = north - south
    #diffx = east - west

    x_points = np.linspace(minx, maxx, num=num_points)
    y_points = np.linspace(miny, maxy, num=num_points)

    network.space_partition = []
    
    #partinioning the city in smaller zones/areas
    #separate this part, because zone partitioning will only happen once, and the calculation of demand units may vary
    for curr_point_x in range(len(x_points)):
        next_point_x = curr_point_x + 1
        if next_point_x < len(x_points):
            for curr_point_y in range(len(y_points)):
                next_point_y = curr_point_y + 1

                if next_point_y < len(y_points):
                    
                    minx = x_points[curr_point_x]
                    miny = y_points[curr_point_y]

                    maxx = x_points[next_point_x]
                    maxy = y_points[next_point_y]

                    #ax.scatter(maxx, maxy, c='blue')
                    #ax.scatter(minx, miny, c='blue')

                    #polygon represents the mini area
                    mini_polygon = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
                    
                    network.space_partition.append(mini_polygon)

    network.units = []

    for origin_polygon in network.space_partition:
        for destination_polygon in network.space_partition:
            d = {
                'origin_polygon': origin_polygon,
                'destination_polygon': destination_polygon,
                'travel_demand': [0] * 24,
            }
            network.units.append(d)


    print('number of units:', len(network.units))

    for request in network.all_requests.values():
        
        hour = int(math.floor(request.get('dep_time')/(3600)))
        
        for i in range(len(network.units)):

            origin_polygon =  network.units[i]['origin_polygon']
            destination_polygon =  network.units[i]['destination_polygon']

            origin_point = Point(request.get('originx'), request.get('originy'))
            destination_point = Point(request.get('destinationx'), request.get('destinationy'))
            
            if origin_polygon.contains(origin_point):
                if destination_polygon.contains(destination_point):
                    network.units[i]['travel_demand'][hour] += 1 
        
    
    print('density threshold:', request_density_treshold)
    #creating graph to cluster
    G_clusters = nx.Graph()
    node_id = 0
    for unit in network.units:
        for hour in range(24):
            if unit['travel_demand'][hour] >= request_density_treshold:
                
                dimensions = []

                uminx, uminy, umaxx, umaxy = unit['origin_polygon'].bounds
                dimensions.append(uminx)
                dimensions.append(uminy)
                origin_point = (uminx, uminy)
                origin_point_inv = (uminy, uminx)
                osmid_origin_walk = ox.get_nearest_node(network.G_walk, origin_point_inv)
                osmid_origin_drive = ox.get_nearest_node(network.G_drive, origin_point_inv)
                #ax.scatter(uminx, uminy, c='red')
                
                uminx, uminy, umaxx, umaxy = unit['destination_polygon'].bounds
                dimensions.append(uminx)
                dimensions.append(uminy)
                destination_point = (uminx, uminy)
                destination_point_inv = (uminy, uminx)
                osmid_destination_walk = ox.get_nearest_node(network.G_walk, destination_point_inv)
                osmid_destination_drive = ox.get_nearest_node(network.G_drive, destination_point_inv)
                #ax.scatter(uminx, uminy, c='red')

                #print(osmid_origin_walk, osmid_destination_walk)

                dimensions.append(hour)

                G_clusters.add_node(node_id, origin_polygon=unit['origin_polygon'], origin_point=origin_point, osmid_origin_walk=osmid_origin_walk, osmid_origin_drive=osmid_origin_drive, destination_polygon=unit['destination_polygon'], destination_point=destination_point, osmid_destination_walk=osmid_destination_walk, osmid_destination_drive=osmid_destination_drive, hour=hour, dimensions=dimensions)
                node_id += 1
    
    for u in G_clusters.nodes():
        for v in G_clusters.nodes():
            
            if u != v:
                adjacent_nodes = check_adj_nodes(G_clusters.nodes[u]['dimensions'], G_clusters.nodes[v]['dimensions'])

                if adjacent_nodes:
                    G_clusters.add_edge(u, v)

    connected_components = nx.connected_components(G_clusters)
    print('connected components:', connected_components)

    stop_locations = []
    for cluster in connected_components:
        if len(cluster) > 1:
            print(cluster)
            print('assign stop locations')
            stop_locations = assign_stop_locations(param, network, cluster, G_clusters)
            print('number of stops', len(stop_locations))
            
            break

    osmid_walk_nodes = []
    osmid_drive_nodes = []
    for stop in stop_locations:
        osmid_walk_nodes.append(stop['osmid_walk'])
        osmid_drive_nodes.append(stop['osmid_drive'])

    stops_folder = os.path.join(param.save_dir_images, 'bus_stops')

    #plot network to show NEW bus stops  
    nc = ['r' if (node in osmid_walk_nodes) else '#336699' for node in network.G_walk.nodes()]
    ns = [12 if (node in osmid_walk_nodes) else 6 for node in network.G_walk.nodes()]
    fig, ax = ox.plot_graph(network.G_walk, show=False, node_size=ns, node_color=nc, node_zorder=2, save=True, filepath=stops_folder+'/_new_stops_walk.png')
        
    #plot network to show NEW bus stops  
    nc = ['r' if (node in osmid_drive_nodes) else '#336699' for node in network.G_drive.nodes()]
    ns = [12 if (node in osmid_drive_nodes) else 6 for node in network.G_drive.nodes()]
    fig, ax = ox.plot_graph(network.G_drive, show=False, node_size=ns, node_color=nc, node_zorder=2, save=True, filepath=stops_folder+'/_new_stops_drive.png')
    #plot the city graph here with the points to see if it is correct
    #center point - green
    #samples - red

    save_dir_csv = os.path.join(param.save_dir, 'csv')
    path_new_bus_stops = os.path.join(save_dir_csv, param.output_file_base+'.new.stops.csv')

    new_stops = pd.DataFrame(stop_locations)
    new_stops.set_index(['stop_id'], inplace=True)
    new_stops.to_csv(path_new_bus_stops)

    travel_time_matrix_new_stops = get_travel_time_matrix_osmnx_csv(param, new_stops, network.shortest_path_drive, network.shortest_path_walk, filename='.travel.time.new.stops.csv')
###start stop locations