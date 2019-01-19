import networkx as nx
import numpy as np

def get_pos_array(graph, latlon=False):
	'''Get numpy array containing the positions of each node in a projected osmnx network.
       If latlon==True, use latidude and longitude coordinates. Otherwise, the projected
       positions are used. Note that if the plot has not been projected, latlon==True will
       give an error and latlon==False will return the latitude and longitude.'''

	nodes = list(graph.nodes(data=True))
	if latlon:
		y = [float(node[1]['lat']) for node in nodes]
		x = [float(node[1]['lon']) for node in nodes]
	else:
		y = [node[1]['y'] for node in nodes]
		x = [node[1]['x'] for node in nodes]
	pos_nodes = np.array([x,y]).T
	return pos_nodes
	
def get_pos_dict(graph, latlon=False):
	'''Get dictionary containing the positions of each node in a projected osmnx network.
       If latlon==True, use latidude and longitude coordinates. Otherwise, the projected
       positions are used. Note that if the plot has not been projected, latlon==True will
       give an error and latlon==False will return the latitude and longitude.'''	

	nodes = list(graph.nodes(data=True))
	if latlon:
		pos_nodes = {node[0]:(float(node[1]['lon']), float(node[1]['lat'])) for node in nodes}
	else:
		pos_nodes = {node[0]:(node[1]['x'], node[1]['y']) for node in nodes}

	return pos_nodes