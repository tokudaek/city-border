import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import networkx.drawing as draw
import cv2
import scipy.ndimage as nd
from skimage.morphology import medial_axis
from matplotlib.patches import Polygon as mpl_polygon
import util

def get_border(graph, sigma, max_ratio, min_width, bin_size=50, pad=1000):
	'''Calculate the urban area border of the input graph. Note that the function expects variable
	   graph to contain two atributes named 'x' and 'y' associated to the spatial position of the nodes.

       Parameters:
       -----------
       graph : networkx graph
           Input graph
       sigma : float
           Standard deviation (in meters) used for the gaussian kernel
       max_ratio : float
           Threshold used for defining high density regions (relative to maximum)
       min_width : float
           Minimum width (in meters) allowed for bottlenecks in the urban area
       bin_size : float
           Size of bins (in meters) used for kernel density estimation
       pad : float
           Padding amount (in meters) to add around city map
		   
       Returns:
       --------
       contour_np : numpy array
           numpy array containing the contour of the urban area.'''

	pos = util.get_pos_array(graph)

	posx, posy = pos.T

	xmin,xmax = np.min(posx),np.max(posx)
	ymin,ymax = np.min(posy),np.max(posy)

	sigma_bins = sigma/bin_size
	min_width_bins = min_width/bin_size

	binsC = np.arange(xmin-pad, xmax+pad+0.5*bin_size, bin_size)
	binsR = np.arange(ymin-pad, ymax+pad+0.5*bin_size, bin_size)
	
	print('Using %d and %d bins on x and y directions'%(len(binsC), len(binsR)))

	hist, _, _ = np.histogram2d(posy, posx,[binsR,binsC])
	G = nd.gaussian_filter(hist, sigma_bins)

	valid_region = G>(max_ratio*np.max(G))

	if (np.sum(valid_region)>0):

		skel, distance = medial_axis(valid_region, return_distance=True)
		dist_on_skel = distance*skel

		ind = np.nonzero(dist_on_skel>min_width_bins)
		r = dist_on_skel[ind]
		r = np.round(r).astype(int)
		region_2 = np.zeros_like(valid_region,dtype=np.uint8)
		for i in range(ind[0].size):
			cv2.circle(region_2, (ind[1][i],ind[0][i]), r[i], 1, -1)
		
		img_lab,num_comp = nd.label(region_2)
		tam_comp = nd.sum(region_2,img_lab,range(1,num_comp+1))
		ind = np.argmax(tam_comp)

		region_f = img_lab==(ind+1)
		
		_,contours,hierarchy = cv2.findContours(region_f.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		contourTemp = [item[0] for item in contours[0]]
		contourTemp = np.array(contourTemp)

		contourTransX = xmin - pad + contourTemp[:,0]*bin_size
		contourTransY = ymin - pad + contourTemp[:,1]*bin_size
		#contourTransX = xmin+(contourTemp[:,0]-pad)*bin_size
		#contourTransY = ymin+(contourTemp[:,1]-pad)*bin_size

		contourTrans = np.array([contourTransX,contourTransY]).T
		contour_np = contourTrans.copy()

	else:	
		contour_np = None
			
	return contour_np

def plot_border(graph, subgraph, border, ax=None):
	'''Plot the original graph, the urban area and the subgraph inside the urban area

       Parameters:
       -----------
       graph : networkx graph
           Original graph
       subgraph : networkx graph
           Subgraph inside the urban area
       border : numpy array
           Array containing the contour of the urban area'''

	# Remove multiedges
	graph = nx.Graph(graph)
	subgraph = nx.Graph(subgraph)

	num_edges = graph.number_of_edges()	
	
	pos = util.get_pos_dict(graph)

	if ax is None:
		fig = plt.figure()
		ax = fig.add_subplot(111, aspect='equal')

	edge_dict = dict(zip(graph.edges, range(num_edges)))
	edge_colors = ['k']*num_edges
	for edge_index, edge in enumerate(graph.edges):
		if edge in subgraph.edges:
			edge_colors[edge_index] = 'r'
	
	draw.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=1)	
	poly = mpl_polygon(border, closed=True, edgecolor='none', facecolor=np.array([0,117,220])/255.,alpha=0.5, antialiased=True)	
	ax.add_patch(poly)	
		
	pos_array = np.array(list(pos.values()))	
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
	ax.set_xlim([np.min(pos_array[:,0]),np.max(pos_array[:,0])])
	ax.set_ylim([np.min(pos_array[:,1]),np.max(pos_array[:,1])])	

	
def extract_subgraph(graph, border):
	'''Extract subgraph inside the contour defined by border. Edges crossing the border are
       also removed. Note that this can create nodes having degree two, which do not represent
       a crossing or termination'''

	pos_dict = util.get_pos_dict(graph)
	pol = mpl_polygon(border, closed=True)
	#pol = Poly(border)
	inside_nodes = []
	for node in graph:
		pos = pos_dict[node]
		if pol.contains_point((pos[0],pos[1])):
		#if pol.isInside(pos[0],pos[1]):
			inside_nodes.append(node)

	subgraph = graph.subgraph(inside_nodes).copy()
	subgraph = max(nx.strongly_connected_component_subgraphs(subgraph, copy=True), key=len)	
	
	return subgraph
	
# Example usage
if __name__=='__main__':

	cityName = "Sao Carlos, SP, Brazil"	

	sigma = 1000		# Standard deviation (in meters) used for the gaussian kernel
	max_ratio = 0.2		# Threshold used for defining high density regions (relative to maximum)
	min_width = 100		# Minimum width (in meters) allowed for bottlenecks in the urban area
	bin_size = 50		# Size of bins (in meters) used for kernel density estimation (lower is better but uses more memory)
	pad = 1000			# Padding amount (in meters) to add around city map

	# Load a graph that was saved using the save_graphml() osmnx function
	cityNetworkProj = ox.load_graphml('%s.graphml'%cityName, '../')
	cityBorder = get_border(cityNetworkProj, sigma, max_ratio, min_width, bin_size, pad)
	
	cityNetworkSubgraph = extract_subgraph(cityNetworkProj, cityBorder)
	
	plot_border(cityNetworkProj, cityNetworkSubgraph, cityBorder)