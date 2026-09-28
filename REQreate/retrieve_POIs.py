import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import os
import osmnx as ox
from .overpass_config import configure_overpass
import pandas as pd
import networkx as nx
import numpy as np
from . import profiling
from . import snap
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    class DummyRay:
        @staticmethod
        def remote(func):
            class RemoteWrapper:
                def remote(*args, **kwargs):
                    return func(*args, **kwargs)
            return RemoteWrapper
        @staticmethod
        def shutdown():
            pass
        @staticmethod
        def init(*args, **kwargs):
            pass
        @staticmethod
        def put(obj):
            return obj
        @staticmethod
        def get(objs):
            return objs
    ray = DummyRay()
import warnings
import gc
from shapely.geometry import Point


def get_POIs_matrix_csv(G_drive, place_name, save_dir, output_folder_base):

    warnings.filterwarnings(action="ignore")
    '''
    retrieve the pois from the location
    '''

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
        
        # These ten queries run back to back, which is what trips Overpass
        # rate limiting; configure_overpass sets the timeout under the name
        # osmnx actually reads and enables the rate limiter.
        #
        # Tempting and tried: merge the five shop queries into one. The tag lists
        # are disjoint (158 values, 158 distinct, none repeated, no other group
        # filters on shop) and osmnx emits one query component per (key, value)
        # pair either way, so one request asks for exactly the same 474 pairs and
        # would cost four fewer rate-limit pauses - worth about 7 minutes.
        #
        # overpass-api.de refuses it. A 20-request Aachen run died after 15
        # minutes with an HTML error page instead of JSON, having answered only
        # the four smaller queries. 474 components each recursing down over a
        # whole city polygon is past what the public instance will do in one
        # request, and the ceiling is a resource limit rather than a size limit,
        # so it scales with the city: any fixed batch size that works for Aachen
        # is still a gamble for somewhere larger. See issue #18.
        #
        # So the queries stay split. The way to stop paying for them is to not
        # make them twice - see REQREATE_CACHE_DIR in overpass_config.
        configure_overpass(timeout=1800)

        pois_shop1 = ox.features_from_place(place_name, tags=tags_shop1)
        print(len(pois_shop1))
        pois_shop2 = ox.features_from_place(place_name, tags=tags_shop2)
        print(len(pois_shop2))
        pois_shop3 = ox.features_from_place(place_name, tags=tags_shop3)
        print(len(pois_shop3))
        pois_shop4 = ox.features_from_place(place_name, tags=tags_shop4)
        print(len(pois_shop4))
        pois_shop5 = ox.features_from_place(place_name, tags=tags_shop5)
        print(len(pois_shop5))

        pois_amenity = ox.features_from_place(place_name, tags=tags_amenity)
        print(len(pois_amenity))
        pois_building = ox.features_from_place(place_name, tags=tags_building)
        print(len(pois_building))
        pois_leisure = ox.features_from_place(place_name, tags=tags_leisure)
        print(len(pois_leisure))
        pois_office = ox.features_from_place(place_name, tags=tags_office)
        print(len(pois_office))
        
        pois_tourism = ox.features_from_place(place_name, tags=tags_tourism)
        print(len(pois_tourism))

        #sum_pois = len(pois_amenity) + len(pois_building) + len(pois_leisure) + len(pois_office) + len(pois_shop1) + len(pois_tourism)
        #print('number pois: ', sum_pois)

        # One nearest_nodes call for every POI in every category, instead of one
        # per POI. ox.nearest_nodes builds a BallTree over all 26,276 nodes of the
        # drive graph on each call, and these ten loops made about 21,000 of them.
        # Concatenating a growing frame once per 100-POI chunk went with it.
        categories = [
            pois_amenity, pois_building, pois_leisure, pois_office,
            pois_shop1, pois_shop2, pois_shop3, pois_shop4, pois_shop5,
            pois_tourism,
        ]
        centroids = [frame.geometry.loc[index].centroid
                     for frame in categories
                     for index in frame.index.tolist()]

        if centroids:
            lons = [c.x for c in centroids]
            lats = [c.y for c in centroids]
            pois = pd.DataFrame({
                'osmid_drive': snap.nearest_nodes(G_drive, lons, lats, 'pois.drive'),
                'lat': lats,
                'lon': lons,
            })

        gc.collect()

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

def _zone_center_distances(network):
    """The zone-centre block of the drive distance matrix, as an n x n array.

    Pulled out in one reindex rather than a .loc per cell. Unreachable pairs
    come back as NaN, which the callers treat as "no contribution" - the same
    outcome the per-cell version produced, since every comparison against a
    missing distance was false.
    """
    centers = network.zones['center_osmid'].to_numpy()
    centers = centers.astype(np.int64)
    block = network.shortest_dist_drive.reindex(index=centers)
    block = block.reindex(columns=[str(c) for c in centers])
    return block.to_numpy(dtype=float)


def _zone_ranks_array(distances, pois):
    """rank[u][v] = POIs in the zones strictly closer to u than v is, excluding u.

    The original computed this with three nested iterrows loops and a .loc per
    innermost step - O(n^3) pandas lookups, which on 272 zones was 20 million
    iterations and about half the runtime of a whole generation.

    Sorting each row by distance turns the inner loop into a prefix sum: the
    POIs closer to u than v is are a prefix of u's row in distance order, found
    with one searchsorted. O(n^2 log n) in numpy instead.
    """
    n = distances.shape[0]
    pois = np.asarray(pois)
    weights = pois.astype(float)
    ranks = np.zeros((n, n), dtype=float)

    for u in range(n):
        row = distances[u]
        order = np.argsort(row, kind='stable')      # NaN sorts to the end
        cumulative = np.concatenate(([0.0], np.cumsum(weights[order])))
        # side='left' counts only entries strictly less than the target, which
        # is the original's `duw < duv` rather than `<=`.
        totals = cumulative[np.searchsorted(row[order], row, side='left')]
        # w == u is excluded by the original's `idw != idu`, and a zone is
        # never strictly closer to itself than v, so drop u's own POIs wherever
        # it would have been counted. w == v needs no guard: d[u,v] < d[u,v] is
        # false, so v never counts itself either.
        totals = totals - np.where(row > row[u], weights[u], 0.0)
        # An unreachable v means no w satisfied duw < duv at all.
        totals[np.isnan(row)] = 0.0
        ranks[u] = totals

    if np.issubdtype(pois.dtype, np.integer):
        return ranks.astype(np.int64)
    return ranks


def _zone_probabilities_array(ranks, alpha):
    """p[u][v] = rank[u][v]**alpha / sum_w rank[w][v]**alpha, zero where rank is 0.

    The original recomputed that denominator inside the u loop, once per u, even
    though it depends only on v - so an O(n^2) quantity cost O(n^3) to build.
    Here it is one column sum.
    """
    ranks = np.asarray(ranks)
    powered = np.zeros(ranks.shape, dtype=float)
    nonzero = ranks != 0
    powered[nonzero] = np.power(ranks[nonzero].astype(float), alpha)

    denominator = powered.sum(axis=0)
    probabilities = np.divide(
        powered, denominator,
        out=np.zeros_like(powered),
        where=denominator != 0,
    )
    # The original wrote 0 for u == v explicitly; rank[u][u] is already 0, so
    # this only restates it.
    np.fill_diagonal(probabilities, 0.0)
    return probabilities


def calc_rank_between_zones(network):

    total_zones = len(network.zones)
    print(f'Finding zone centers: {total_zones} zones, in one query')
    centers = snap.nearest_nodes(network.G_drive,
                                 network.zones['center_x'].tolist(),
                                 network.zones['center_y'].tolist(),
                                 'zones.centers')
    # float, because the column used to be seeded with NaN and then filled cell by
    # cell, which left it float64 - and that dtype reaches the zones CSV.
    network.zones['center_osmid'] = np.asarray(centers, dtype=float)

    print('Calculating zone ranks...')

    with profiling.stage('zones.rank_between_zones'):
        zone_ranks = pd.DataFrame(
            _zone_ranks_array(
                _zone_center_distances(network),
                network.zones['number_pois'].to_numpy(),
            ),
            index=network.zones.index,
            columns=network.zones.index,
        )
    zone_ranks.index.name = 'zone_id'
    zone_ranks = zone_ranks.reset_index()
    #save_dir_csv = os.path.join(save_dir, 'csv')
    #path_pois_file = os.path.join(save_dir_csv, place_name+'.pois.csv') 
    #zone_ranks.to_csv(path_pois_file)
    zone_ranks.set_index(['zone_id'], inplace=True)
    #print(zone_ranks.head())
    #zone_ranks["sum"] = zone_ranks.sum(axis=1)

    return zone_ranks

def calc_probability_travel_between_zones(network, zone_ranks, alpha):

    alpha = alpha*-1

    with profiling.stage('zones.probability_between_zones'):
        zone_probabilities = pd.DataFrame(
            _zone_probabilities_array(zone_ranks.to_numpy(), alpha),
            index=zone_ranks.index,
            columns=zone_ranks.columns,
        )
    zone_probabilities.index.name = 'zone_id'
    zone_probabilities = zone_probabilities.reset_index()
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
