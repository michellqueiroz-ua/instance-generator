import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import os
import osmnx as ox
import pandas as pd
import networkx as nx
import numpy as np
import ray
import warnings
import gc
from shapely.geometry import Point


def divide_chunks(l, n):
      
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

@ray.remote
def get_poi(G_drive, index, all_pois):

    poi = all_pois.loc[index]
    poi_point = (poi.geometry.centroid.y, poi.geometry.centroid.x)
    
    #u, v, key = ox.get_nearest_edge(G_walk, poi_point)
    #poi_node_walk = min((u, v), key=lambda n: ox.distance.great_circle_vec(poi.geometry.centroid.y, poi.geometry.centroid.x, G_walk.nodes[n]['y'], G_walk.nodes[n]['x']))
    
    #u, v, key = ox.get_nearest_edge(G_drive, poi_point)
    #poi_node_drive = min((u, v), key=lambda n: ox.distance.great_circle_vec(poi.geometry.centroid.y, poi.geometry.centroid.x, G_drive.nodes[n]['y'], G_drive.nodes[n]['x']))
    
    poi_node_drive = ox.nearest_nodes(G_drive, poi_point[1], poi_point[0])

    d = {
        #'station_id': index,
        #'osmid_walk': poi_node_walk,
        'osmid_drive': poi_node_drive,
        'lat': poi.geometry.centroid.y,
        'lon': poi.geometry.centroid.x
    }

    return d

def get_POIs_matrix_csv(G_drive, place_name, save_dir, output_folder_base):

    warnings.filterwarnings(action="ignore")
    '''
    retrieve the pois from the location
    '''

    ray.shutdown()
    ray.init(num_cpus=cpu_count())

    save_dir_csv = os.path.join(save_dir, 'csv')

    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)

    pois = pd.DataFrame()
    path_pois = os.path.join(save_dir_csv, output_folder_base+'.pois.csv')

    if os.path.isfile(path_pois):
        print('is file POIs')
        pois = pd.read_csv(path_pois)
        
    else:
        print('creating file POIs')

        #retrieve pois
        tags_amenity = {
            'amenity':['bar','cafe','restaurant','pub','fast_food','college','driving_school','kindergarten','language_school','library','toy_library','school','university','bicycle_rental','baby_hatch','clinic','dentist','doctors','hospital','nursing_home','social_facility','veterinary','arts_centre','brothel','casino','cinema','community_centre','conference_centre','events_venue','gambling','love_hotel','night_club','planetarium','social_centre','stripclub','swingerclub','theatre','courthouse','embassy','police','ranger_station','townhall','internet_cafe','marketplace'],
            }

        tags_building = {
            'building':['apartments','detached','dormitory','hotel','house','residential','semidetached_house','commercial','industrial','office','retail','supermarket','warehouse','cathedral','chapel','church','monastery','mosque','government','train_station','stadium'],
            'historic':'castle',
            }

        tags_leisure = {
            'leisure':['fitness_centre','park','sports_centre','swimming_pool','stadium'],
            'man_made':['obelisk','observatory'],
        }

        tags_office = {
            'office':['accountant','advertising_agency','architect','charity','company','consulting','courier','coworking','educational_institution','employment_agency','engineer','estate_agent','financial','financial_advisor','forestry','foundation','government','graphic_design','insurance','it','lawyer','logistics','moving_company','newspaper','ngo','political_party','property_management','research','tax_advisor','telecommunication','visa','water_utility'],
        }

        tags_shop1 = {
            'shop':['alcohol','bakery','beverages','brewing_supplies','butcher','cheese','chocolate','coffee','confectionery','convenience','deli','dairy','farm','frozen_food','greengrocer','health_food','ice_cream','pasta','pastry','spices','tea','department_store','general','kiosk','mall','supermarket','baby_goods','bag','boutique','clothes','fabric'],
        }

        tags_shop2 = {
            'shop':['fashion_accessories','jewelry','leather','sewing','shoes','tailor','watches','wool','charity','second_hand','variety_store','beauty','chemist','cosmetics','erotic','hairdresser','hairdresser_supply','hearing_aids','herbalist','massage','medical_supply','nutrition_supplements','optician','perfumery','tattoo','agrarian','appliance','bathroom_furnishing','doityourself','electrical','energy','fireplace','florist'],
        }

        tags_shop3 = {
            'shop':['garden_centre','garden_furniture','gas','glaziery','groundskeeping','hardware','houseware','locksmith','paint','security','trade','antiques','bed','candles','carpet','curtain','doors','flooring','furniture','household_linen','interior_decoration','kitchen','lighting','tiles','window_blind','computer','electronics','hifi','mobile_phone','radiotechnics','vacuum_cleaner','atv','bicycle','boat','car','car_repair'],
        }

        tags_shop4 = {
            'shop':['car_parts','caravan','fuel','fishing','golf','hunting','jetski','military_surplus','motorcycle','outdoor','scuba_diving','ski','snowmobile','sports','swimming_pool','trailer','tyres','art','collector','craft','frame'],
        }

        tags_shop5 = {
            'shop':['games','model','music','musical_instrument','photo','camera','trophy','video','video_games','anime','books','gift','lottery','newsagent','stationery','ticket','bookmaker','cannabis','copyshop','dry_cleaning','e-cigarette','funeral_directors','laundry','money_lender','party','pawnbroker','pet','pet_grooming','pest_control','pyrotechnics','religion','storage_rental','tobacco','toys','travel_agency','weapons','outpost'],
        }

        tags_tourism = { 
            'tourism':['aquarium','artwork','attraction','gallery','hostel','motel','museum','theme_park','zoo'],
        }
        
        ox.settings.timeout = 1800

        pois_shop1 = ox.geometries_from_place(place_name, tags=tags_shop1)
        print(len(pois_shop1))
        pois_shop2 = ox.geometries_from_place(place_name, tags=tags_shop2)
        print(len(pois_shop2))
        pois_shop3 = ox.geometries_from_place(place_name, tags=tags_shop3)
        print(len(pois_shop3))
        pois_shop4 = ox.geometries_from_place(place_name, tags=tags_shop4)
        print(len(pois_shop4))
        pois_shop5 = ox.geometries_from_place(place_name, tags=tags_shop5)
        print(len(pois_shop5))

        pois_amenity = ox.geometries_from_place(place_name, tags=tags_amenity)
        print(len(pois_amenity))
        pois_building = ox.geometries_from_place(place_name, tags=tags_building)
        print(len(pois_building))
        pois_leisure = ox.geometries_from_place(place_name, tags=tags_leisure)
        print(len(pois_leisure))
        pois_office = ox.geometries_from_place(place_name, tags=tags_office)
        print(len(pois_office))
        
        pois_tourism = ox.geometries_from_place(place_name, tags=tags_tourism)
        print(len(pois_tourism))

        #sum_pois = len(pois_amenity) + len(pois_building) + len(pois_leisure) + len(pois_office) + len(pois_shop1) + len(pois_tourism)
        #print('number pois: ', sum_pois)

        #G_walk_id = ray.put(G_walk)
        G_drive_id = ray.put(G_drive)
        pois_amenity_id = ray.put(pois_amenity)
        pois_building_id = ray.put(pois_building)
        pois_leisure_id = ray.put(pois_leisure)
        pois_office_id = ray.put(pois_office)
        pois_shop_id1 = ray.put(pois_shop1)
        pois_shop_id2 = ray.put(pois_shop2)
        pois_shop_id3 = ray.put(pois_shop3)
        pois_shop_id4 = ray.put(pois_shop4)
        pois_shop_id5 = ray.put(pois_shop5)
        pois_tourism_id = ray.put(pois_tourism)
        chunksize = 100

        pois_amenity_index = pois_amenity.index.tolist()
        chunks_pois = list(divide_chunks(pois_amenity_index, chunksize))
        count = 0
        for cpois in chunks_pois:
            #print(count)
            poisx = ray.get([get_poi.remote(G_drive_id, index_poi, pois_amenity_id) for index_poi in cpois]) 
            count += 1

            xt = pd.DataFrame(poisx)
            pois = pois.append(xt, ignore_index=True)
            
            gc.collect()

        pois_building_index = pois_building.index.tolist()
        chunks_pois = list(divide_chunks(pois_building_index, chunksize))
        count = 0
        for cpois in chunks_pois:
            #print(count)
            poisx = ray.get([get_poi.remote(G_drive_id, index_poi, pois_building_id) for index_poi in cpois]) 
            count += 1

            xt = pd.DataFrame(poisx)
            pois = pois.append(xt, ignore_index=True)
            
            gc.collect()

        pois_leisure_index = pois_leisure.index.tolist()
        chunks_pois = list(divide_chunks(pois_leisure_index, chunksize))
        count = 0
        for cpois in chunks_pois:
            #print(count)
            poisx = ray.get([get_poi.remote(G_drive_id, index_poi, pois_leisure_id) for index_poi in cpois]) 
            count += 1

            xt = pd.DataFrame(poisx)
            pois = pois.append(xt, ignore_index=True)
            
            gc.collect()

        pois_office_index = pois_office.index.tolist()
        chunks_pois = list(divide_chunks(pois_office_index, chunksize))
        count = 0
        for cpois in chunks_pois:
            #print(count)
            poisx = ray.get([get_poi.remote(G_drive_id, index_poi, pois_office_id) for index_poi in cpois]) 
            count += 1

            xt = pd.DataFrame(poisx)
            pois = pois.append(xt, ignore_index=True)
            
            gc.collect()

        pois_shop_index1 = pois_shop1.index.tolist()
        chunks_pois = list(divide_chunks(pois_shop_index1, chunksize))
        count = 0
        for cpois in chunks_pois:
            #print(count)
            poisx = ray.get([get_poi.remote(G_drive_id, index_poi, pois_shop_id1) for index_poi in cpois]) 
            count += 1

            xt = pd.DataFrame(poisx)
            pois = pois.append(xt, ignore_index=True)
            
            gc.collect()

        pois_shop_index2 = pois_shop2.index.tolist()
        chunks_pois = list(divide_chunks(pois_shop_index2, chunksize))
        count = 0
        for cpois in chunks_pois:
            #print(count)
            poisx = ray.get([get_poi.remote(G_drive_id, index_poi, pois_shop_id2) for index_poi in cpois]) 
            count += 1

            xt = pd.DataFrame(poisx)
            pois = pois.append(xt, ignore_index=True)
            
            gc.collect()

        pois_shop_index3 = pois_shop3.index.tolist()
        chunks_pois = list(divide_chunks(pois_shop_index3, chunksize))
        count = 0
        for cpois in chunks_pois:
            #print(count)
            poisx = ray.get([get_poi.remote(G_drive_id, index_poi, pois_shop_id3) for index_poi in cpois]) 
            count += 1

            xt = pd.DataFrame(poisx)
            pois = pois.append(xt, ignore_index=True)
            
            gc.collect()

        pois_shop_index4 = pois_shop4.index.tolist()
        chunks_pois = list(divide_chunks(pois_shop_index4, chunksize))
        count = 0
        for cpois in chunks_pois:
            #print(count)
            poisx = ray.get([get_poi.remote(G_drive_id, index_poi, pois_shop_id4) for index_poi in cpois]) 
            count += 1

            xt = pd.DataFrame(poisx)
            pois = pois.append(xt, ignore_index=True)
            
            gc.collect()

        pois_shop_index5 = pois_shop5.index.tolist()
        chunks_pois = list(divide_chunks(pois_shop_index5, chunksize))
        count = 0
        for cpois in chunks_pois:
            #print(count)
            poisx = ray.get([get_poi.remote(G_drive_id, index_poi, pois_shop_id5) for index_poi in cpois]) 
            count += 1

            xt = pd.DataFrame(poisx)
            pois = pois.append(xt, ignore_index=True)
            
            gc.collect()

        pois_tourism_index = pois_tourism.index.tolist()
        chunks_pois = list(divide_chunks(pois_tourism_index, chunksize))
        count = 0
        for cpois in chunks_pois:
            #print(count)
            poisx = ray.get([get_poi.remote(G_drive_id, index_poi, pois_tourism_id) for index_poi in cpois]) 
            count += 1

            xt = pd.DataFrame(poisx)
            pois = pois.append(xt, ignore_index=True)
            
            gc.collect()
        
        
        ray.shutdown()

        pois = pd.DataFrame(pois)

        pois.to_csv(path_pois)
        
    return pois

def plot_pois(network, save_dir_images):

    '''
    create figures with the POIs present in the location
    '''

    pois_folder = os.path.join(save_dir_images, 'pois')

    if not os.path.isdir(pois_folder):
        os.mkdir(pois_folder)

    pois_nodes = []
    for index, poi in network.pois.iterrows():
        pois_nodes.append(poi['osmid_walk'])

    nc = ['#FF0000' if (node in pois_nodes) else '#000000' for node in network.G_walk.nodes()]
    ns = [20 if (node in pois_nodes) else 12 for node in network.G_walk.nodes()]
    fig, ax = ox.plot_graph(network.G_walk, node_size=ns, figsize=(8, 8), show=False, bgcolor="#ffffff", node_color=nc, node_zorder=2, save=True,edge_color="#999999", edge_alpha=None, dpi=1440, filepath=pois_folder+'/pois_walk.png')
    

    plt.close(fig)

    pois_nodes = []
    for index, poi in network.bus_stations.iterrows():
        pois_nodes.append(poi['osmid_drive'])

    nc = ['#FF0000' if (node in pois_nodes) else '#000000' for node in network.G_drive.nodes()]
    ns = [20 if (node in pois_nodes) else 20 for node in network.G_drive.nodes()]
    fig, ax = ox.plot_graph(network.G_drive, node_size=ns, figsize=(8, 8), show=False, bgcolor="#ffffff", node_color=nc, node_zorder=2,edge_color="#999999", edge_alpha=None, dpi=1440, filepath=pois_folder+'/pois_drive.png')
    
    #for poi in pois_nodes:

    #    ax.scatter(network.G_drive.nodes[poi]['x'], network.G_drive.nodes[poi]['y'], c='black', s=60, marker=",")
    
    plt.savefig(pois_folder+'/pois_drive.png')
    plt.close(fig)

def attribute_density_zones(network, pois):

    network.zones['number_pois'] = 0
    
    for idx2, poi in pois.iterrows():
        #pnt = (poi['lat'], poi['lon'])
        pnt = Point(poi['lon'], poi['lat'])
        for idx, zone in network.zones.iterrows():

            zone_polygon = zone['polygon']
        
            if zone_polygon.contains(pnt): 

                network.zones.loc[idx, 'number_pois'] = zone['number_pois'] + 1
                break

    total_sum = network.zones['number_pois'].sum()
    print('total_sum: ', total_sum)

    network.zones['density_pois'] = (network.zones['number_pois']/total_sum)*100
    print(network.zones['density_pois'].head())
    #print(network.zones['density_pois'].sum())

def calc_rank_between_zones(network):

    network.zones['center_osmid'] = np.nan
    for idx, zone in network.zones.iterrows():

        center_point = (zone['center_y'], zone['center_x'])
        center_osmid = ox.nearest_nodes(network.G_drive, center_point[1], center_point[0])
        network.zones.loc[idx, 'center_osmid'] = int(center_osmid)

    zone_ranks = []
    
    for idu, zoneu in network.zones.iterrows():
        rank = {}
        rank['zone_id'] = idu
        for idv, zonev in network.zones.iterrows():
            rank[idv] = 0
            duv = network.shortest_dist_drive.loc[int(zoneu['center_osmid']), str(int(zonev['center_osmid']))]
            for idw, zonew in network.zones.iterrows():

                if ((idw != idu) and (idw != idv)):
                    duw = network.shortest_dist_drive.loc[int(zoneu['center_osmid']), str(int(zonew['center_osmid']))]
                    if duw < duv:
                        rank[idv] += network.zones.loc[idw, 'number_pois']

        zone_ranks.append(rank)
        del rank

    zone_ranks = pd.DataFrame(zone_ranks)  
    #save_dir_csv = os.path.join(save_dir, 'csv')
    #path_pois_file = os.path.join(save_dir_csv, place_name+'.pois.csv') 
    #zone_ranks.to_csv(path_pois_file)
    zone_ranks.set_index(['zone_id'], inplace=True)
    #print(zone_ranks.head())
    #zone_ranks["sum"] = zone_ranks.sum(axis=1)

    return zone_ranks

def calc_probability_travel_between_zones(network, zone_ranks, alpha):

    zone_probabilities = []

    alpha = alpha*-1

    for idu, zoneu in zone_ranks.iterrows():
        puv = {}
        puv['zone_id'] = idu
        for idv, zonev in zone_ranks.iterrows():
            
            if idu != idv:
                p1 = zone_ranks.loc[int(idu), int(idv)]

                r2 = 0
                for idw, zonew in zone_ranks.iterrows():
                    p2 = zone_ranks.loc[int(idw), int(idv)]
                    if p2 != 0:
                        r2 += p2 ** alpha

                if p1 != 0:
                    r1 = p1 ** alpha
                    
                    puv[idv] = r1/r2
                else:
                    puv[idv] = 0
            else:
                puv[idv] = 0


        zone_probabilities.append(puv)
        del puv
        gc.collect()

    zone_probabilities = pd.DataFrame(zone_probabilities)  
    zone_probabilities.set_index(['zone_id'], inplace=True)

    return zone_probabilities
    #zone_probabilities["sum"] = zone_probabilities.sum(axis=1)

    #for idx, zone in zone_probabilities.iterrows():

        #print(idx, ": ", zone['sum'])

        #print(idx, ": ", zone_probabilities[int(idx)].sum())
 
@ray.remote
def get_zone(zones, pt):   

    for idx, zone in zones.iterrows():

        zone_polygon = zone['polygon']
    
        if zone_polygon.contains(pt): 

            return idx

    return np.nan

def rank_of_displacements(network, zone_ranks, df):


    df['zone_origin'] = np.nan
    df['zone_destination'] = np.nan

    x, chunksize = 1, 100000
    for dfc in np.array_split(df, 100):

        ray.shutdown()
        ray.init(num_cpus=cpu_count())
        zones_id = ray.put(network.zones)
        zone_origins = ray.get([get_zone.remote(zones_id, Point(trip['Pickup_Centroid_Longitude'], trip['Pickup_Centroid_Latitude'])) for idx2, trip in dfc.iterrows()]) 

        del zones_id
        gc.collect()

        j = 0
        for idx2, trip in dfc.iterrows():
    
            df.loc[idx2, 'zone_origin'] = zone_origins[j]
            j += 1

        del zone_origins
        gc.collect()

    x, chunksize = 1, 100000
    for dfc in np.array_split(df, 100):

        ray.shutdown()
        ray.init(num_cpus=cpu_count())
        zones_id = ray.put(network.zones)
        zone_destinations = ray.get([get_zone.remote(zones_id, Point(trip['Dropoff_Centroid_Longitude'], trip['Dropoff_Centroid_Latitude'])) for idx2, trip in dfc.iterrows()]) 
        
        del zones_id
        gc.collect()

        j = 0
        for idx2, trip in dfc.iterrows():
    
            df.loc[idx2, 'zone_destination'] = zone_destinations[j]
            j += 1

        del zone_destinations
        gc.collect()

    #print(zone_origins)
    
    print(df['zone_origin'].head()) 
    print(df['zone_destination'].head()) 

    df.dropna(subset=['zone_origin'], inplace=True)
    df.dropna(subset=['zone_destination'], inplace=True)

    df['rank_trip'] = np.nan
    for idx, trip in df.iterrows():

        idu = int(trip['zone_origin'])
        idv = int(trip['zone_destination'])
        #print(zone_ranks.loc[idu, idv])
        df.loc[idx, 'rank_trip'] = float(zone_ranks.loc[idu, idv])

    print(df['rank_trip'].head())
    df.dropna(subset=['rank_trip'], inplace=True)

    gc.collect()
    ray.shutdown()

    return df
