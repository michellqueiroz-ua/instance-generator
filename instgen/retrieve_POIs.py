import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import os
import osmnx as ox
import pandas as pd
import ray
import warnings
import gc
from shapely.geometry import Point


def divide_chunks(l, n):
      
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

@ray.remote
def get_poi(G_walk, G_drive, index, all_pois):

    poi = all_pois.loc[index]
    poi_point = (poi.geometry.centroid.y, poi.geometry.centroid.x)
    
    #u, v, key = ox.get_nearest_edge(G_walk, poi_point)
    #poi_node_walk = min((u, v), key=lambda n: ox.distance.great_circle_vec(poi.geometry.centroid.y, poi.geometry.centroid.x, G_walk.nodes[n]['y'], G_walk.nodes[n]['x']))
    
    #u, v, key = ox.get_nearest_edge(G_drive, poi_point)
    #poi_node_drive = min((u, v), key=lambda n: ox.distance.great_circle_vec(poi.geometry.centroid.y, poi.geometry.centroid.x, G_drive.nodes[n]['y'], G_drive.nodes[n]['x']))
    
    poi_node_drive = ox.get_nearest_node(G_drive, poi_point)

    d = {
        #'station_id': index,
        #'osmid_walk': poi_node_walk,
        'osmid_drive': poi_node_drive,
        'lat': poi.geometry.centroid.y,
        'lon': poi.geometry.centroid.x
    }

    return d

def get_POIs_matrix_csv(G_walk, G_drive, place_name, save_dir, output_folder_base):

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
        tags = {
            'amenity':['bar','cafe','restaurant','pub','fast_food','college','driving_school','kindergarten','language_school','library','toy_library','school','university','bicycle_rental','baby_hatch','clinic','dentist','doctors','hospital','nursing_home','social_facility','veterinary','arts_centre','brothel','casino','cinema','community_centre','conference_centre','events_venue','gambling','love_hotel','night_club','planetarium','social_centre','stripclub','swingerclub','theatre','courthouse','embassy','police','ranger_station','townhall','internet_cafe','marketplace'],
            }
        
        '''   
            'building':['apartments','detached','dormitory','hotel','house','residential','semidetached_house','commercial','industrial','office','retail','supermarket','warehouse','cathedral','chapel','church','monastery','mosque','government','train_station','stadium'],
                    'historic':'castle',
            'leisure':['fitness_centre','park','sports_centre','swimming_pool','stadium'],
            'man_made':['obelisk','observatory'],
        tags2 = { 
            'office':['accountant','advertising_agency','architect','charity','company','consulting','courier','coworking','educational_institution','employment_agency','engineer','estate_agent','financial','financial_advisor','forestry','foundation','government','graphic_design','insurance','it','lawyer','logistics','moving_company','newspaper','ngo','political_party','property_management','research','tax_advisor','telecommunication','visa','water_utility'],
            'shop':'alcohol',
            'shop':'bakery', 
            'shop':'beverages', 
            'shop':'brewing_supplies', 
            'shop':'butcher', 
            'shop':'cheese', 
            'shop':'chocolate', 
            'shop':'coffee', 
            'shop':'confectionery', 
            'shop':'convenience', 
            'shop':'deli', 
            'shop':'dairy', 
            'shop':'farm', 
            'shop':'frozen_food',
            'shop':'greengrocer', 
            'shop':'health_food', 
            'shop':'ice_cream', 
            'shop':'pasta', 
            'shop':'pastry', 
            'shop':'spices', 
            'shop':'tea', 
            'shop':'department_store', 
            'shop':'general', 
            'shop':'kiosk', 
            'shop':'mall',  
            'shop':'supermarket', 
            'shop':'baby_goods', 
            'shop':'bag', 
            'shop':'boutique', 
            'shop':'clothes', 
            'shop':'fabric', 
            'shop':'fashion_accessories',
            'shop':'jewelry',
            'shop':'leather',
            'shop':'sewing',
            'shop':'shoes',
            'shop':'tailor',
            'shop':'watches',      
            'shop':'wool',
            'shop':'charity',
            'shop':'second_hand',
            'shop':'variety_store',
            'shop':'beauty',
            'shop':'chemist',
            'shop':'cosmetics',
            'shop':'erotic',
            'shop':'hairdresser',
            'shop':'hairdresser_supply',
            'shop':'hearing_aids',
            'shop':'herbalist',
            'shop':'massage',
            'shop':'medical_supply',
            'shop':'nutrition_supplements',
            'shop':'optician',
            'shop':'perfumery',
            'shop':'tattoo',
            'shop':'agrarian',
            'shop':'appliance',
            'shop':'bathroom_furnishing',
            'shop':'doityourself',
            'shop':'electrical',
            'shop':'energy',
            'shop':'fireplace',
            'shop':'florist',
            'shop':'garden_centre',
            'shop':'garden_furniture',
            'shop':'gas',
            'shop':'glaziery',
            'shop':'groundskeeping',
            'shop':'hardware',
            'shop':'houseware',
            'shop':'locksmith',
            'shop':'paint',
            'shop':'security',
            'shop':'trade',
            'shop':'antiques',
            'shop':'bed',
            'shop':'candles',
            'shop':'carpet',
            'shop':'curtain',
            'shop':'doors',
            'shop':'flooring',
            'shop':'furniture',
            'shop':'household_linen',
            'shop':'interior_decoration',
            'shop':'kitchen',
            'shop':'lighting',
            'shop':'tiles',
            'shop':'window_blind',
            'shop':'computer',
            'shop':'electronics',
            'shop':'hifi',
            'shop':'mobile_phone',
            'shop':'radiotechnics',
            'shop':'vacuum_cleaner',
            'shop':'atv',
            'shop':'bicycle',
            'shop':'boat',
            'shop':'car',
            'shop':'car_repair',
            'shop':'car_parts',
            'shop':'caravan',
            'shop':'fuel',
            'shop':'fishing',
            'shop':'golf',
            'shop':'hunting',
            'shop':'jetski',
            'shop':'military_surplus',
            'shop':'motorcycle',
            'shop':'outdoor',
            'shop':'scuba_diving',
            'shop':'ski',
            'shop':'snowmobile',
            'shop':'sports',
            'shop':'swimming_pool',
            'shop':'trailer',
            'shop':'tyres',
            'shop':'art',
            'shop':'collector',
            'shop':'craft',
            'shop':'frame',
            'shop':'games',
            'shop':'model',
            'shop':'music',
            'shop':'musical_instrument',
            'shop':'photo',
            'shop':'camera',
            'shop':'trophy',
            'shop':'video',
            'shop':'video_games',
            'shop':'anime',
            'shop':'books',
            'shop':'gift',
            'shop':'lottery',
            'shop':'newsagent',
            'shop':'stationery',
            'shop':'ticket',
            'shop':'bookmaker',
            'shop':'cannabis',
            'shop':'copyshop',
            'shop':'dry_cleaning',
            'shop':'e-cigarette',
            'shop':'funeral_directors',
            'shop':'laundry',
            'shop':'money_lender',
            'shop':'party',
            'shop':'pawnbroker',
            'shop':'pet',
            'shop':'pet_grooming',
            'shop':'pest_control',
            'shop':'pyrotechnics',
            'shop':'religion',
            'shop':'storage_rental',
            'shop':'tobacco',
            'shop':'toys',
            'shop':'travel_agency',
            'shop':'weapons',
            'shop':'outpost',
            'tourism':'aquarium',
            'tourism':'artwork',
            'tourism':'attraction',
            'tourism':'gallery',
            'tourism':'hostel',
            'tourism':'motel',
            'tourism':'museum',
            'tourism':'theme_park',
            'tourism':'zoo',
        }
        '''

        all_pois = ox.geometries_from_place(place_name, tags=tags)

        print('number pois: ', len(all_pois))
        G_walk_id = ray.put(G_walk)
        G_drive_id = ray.put(G_drive)

        all_pois_index = all_pois.index.tolist()
        chunksize = 100
        chunks_pois = list(divide_chunks(all_pois_index, chunksize))
        
        count = 0
        for cpois in chunks_pois:
            print(count)
            poisx = ray.get([get_poi.remote(G_walk_id, G_drive_id, index_poi, all_pois) for index_poi in cpois]) 
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

def attribute_density_zones(inst, pois):

    
    inst.network.zones['number_pois'] = 0
    
    for idx2, poi in pois.iterrows():
        #pnt = (poi['lat'], poi['lon'])
        pnt = Point(poi['lon'], poi['lat'])
        for idx, zone in inst.network.zones.iterrows():

            zone_polygon = zone['polygon']
        
            if zone_polygon.contains(pnt): 

                inst.network.zones.loc[idx, 'number_pois'] = zone['number_pois'] + 1
                break

    total_sum = inst.network.zones['number_pois'].sum()
    print('total_sum: ', total_sum)

    inst.network.zones['density_pois'] = (inst.network.zones['number_pois']/total_sum)*100
    print(inst.network.zones['density_pois'].head())

#def inter_opp_zones(zones, idz):

#    for idx2, zone2 in inst.network.zones.iterrows():

def calc_rank_between_zones(inst):

    inst.network.zones['center_osmid'] = np.nan
    for idx, zone in inst.network.zones.iterrows():

        center_point = (zone['center_y'], zone['center_x'])
        center_osmid = ox.get_nearest_node(inst.network.G_drive, center_point)
        inst.network.zones.loc[idx, 'center_osmid'] = int(center_osmid)

    zone_ranks = []
    
    for idu, zoneu in inst.network.zones.iterrows():
        rank = {}
        rank['zone_id'] = idu
        for idv, zonev in inst.network.zones.iterrows():
            rank[idv] = 0
            duv = inst.network.shortest_dist_drive.loc[int(zoneu['center_osmid']), str(zonev['center_osmid'])]
            for idw, zonew in inst.network.zones.iterrows():

                if ((idw != idu) and (idw != idv)):
                    duw = inst.network.shortest_dist_drive.loc[int(zoneu['center_osmid']), str(zonew['center_osmid'])]
                    if duw < duv:
                        rank[idv] += inst.network.zones.loc[idw, 'number_pois']

        zone_ranks.append(rank)
        del rank

    zone_ranks = pd.DataFrame(zone_ranks)  
    #save_dir_csv = os.path.join(save_dir, 'csv')
    #path_pois_file = os.path.join(save_dir_csv, place_name+'.pois.csv') 
    #zone_ranks.to_csv(path_pois_file)
    zone_ranks.set_index(['zone_id'], inplace=True)
    #zone_ranks["sum"] = zone_ranks.sum(axis=1)

    return zone_ranks

def calc_probability_travel_between_zones(inst, zone_ranks, alpha):

    zone_probabilities = []

    alpha = alpha*-1

    for idu, zoneu in zone_ranks.iterrows():
        puv = {}
        puv['zone_id'] = idu
        for idv, zonev in zone_ranks.iterrows():

            p1 = zone_ranks.loc[int(idu), str(idv)]
            p2 = zone_ranks[str(idv)].sum()

            r1 = p1 ** alpha
            r2 = p2 ** alpha
            puv[idv] = r1/r2

        zone_probabilities.append(puv)
        del puv

    zone_probabilities = pd.DataFrame(zone_probabilities)  
    zone_probabilities.set_index(['zone_id'], inplace=True)

    zone_probabilities["sum"] = zone_probabilities.sum(axis=1)

    for idx, zone in zone_probabilities.iterrows():

        print(idx, ": ", zone['sum'])
        