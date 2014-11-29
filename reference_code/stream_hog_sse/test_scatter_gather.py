from math import floor
from math import ceil
from IPython import embed
'''
Rethinking computeCells as a 'gather' instead of 'scatter' operation.
This script is to verify that our indexing in 'gather' matches the original 'scatter' code.
'''

#pixel -> ixp (gold code from voc-release5)
#    ixp = location in output histogram
def ixp_scatter(pixel, sbin):
    xp = (float(pixel)+0.5)/(float(sbin)) - 0.5;
    ixp = int(floor(xp))
    #vx0 = xp-ixp
    #return [ixp, vx0]
    return [ixp, ixp+1] #locations in output histogram
    #TODO: vx1, ixp+1?

#pixel -> vx0 
#    vx0 = one of the weights for writing to histogram
def px_to_vx0(pixel, sbin):
    xp = (float(pixel)+0.5)/(float(sbin)) - 0.5;
    ixp = int(floor(xp))
    vx0 = xp-ixp
    return [ixp, vx0]

#ixp -> pixels in this bin
def px_gather(ixp, sbin):
    #ixp and sbin should both be int.
    sbin = int(sbin)
    ixp = int(ixp)

    #minPx = max(0, ixp*sbin - sbin/2 + 1)
    #maxPx = min(ixp*sbin + 2*sbin - 2, imgWidth-1)

    #without bounds checking (for this test)
    #minPx = ixp*sbin - sbin/2 + 1
    #maxPx = ixp*sbin + 2*sbin - 2

    #3rd try...
    #   works for sbin=4, sbin=6, sbin=8. off-by-one (high) for sbin=7.
    #minPx = ixp*sbin - sbin/2
    #maxPx = ixp*sbin + 2*sbin - sbin/2 - 1

    #4th try...
    #   works for sbin=4, 6, 7 
    minPx = ixp*sbin - int(ceil(float(sbin)/2)) 
    maxPx = ixp*sbin + 2*sbin - int(ceil(float(sbin)/2)) - 1

    px_range = list(xrange(minPx, maxPx+1))
    return px_range
    #TODO: vx0?

#"I'm a bin, I found a pixel, how do I weight it?"
#idx_in_bin -- in range of (0 to #elements contributing to bin)
def get_vx0(idx_in_bin, sbin):

    #making this up as I go...
    dist_from_center = abs(idx_in_bin - sbin) + 1
    print dist_from_center
    vx0 = 1.0 - ( (dist_from_center - 0.5)/4.0 ) 
    return vx0

    '''
    #vx0_range = list(xrange(1.5, sbin+0.5)) + list(xrange(sbin+0.5, 1.5, -1)) 
    vx0_range = np.array(sbin/2 - 1) 
    
    '''

def test_get_vx0(sbin):
    print "***test_get_vx0()"
    #sbin=4
    vx0 = []
    for i in xrange(0, sbin*2):
        vx0.append(get_vx0(i, sbin))
    print vx0

#lookup table of weights for pixels to be added to a hog cell.
def init_v_LUT(sbin):

    v_LUT = []
    #preallocate.
    for i in xrange(0, sbin*2):
        v_LUT.append(0)

    #e.g. (i=1) -> v_LUT[i]=3/8
    for i in xrange(0, sbin):
        #e.g. for sbin=4, v_LUT = [1/8, 3/8, 5/8, 7/8, 7/8, 5/8, 3/8, 1/8]
        if sbin%2 == 0:
            v_LUT[i] = (i*2 + 1.0)/(sbin*2.0) #TODO: check if floating-pt is needed for odd values of sbin
        if sbin%2 == 1: #special case for odd (rarely if ever used)
            v_LUT[i] = (i*2)/(sbin*2.0)
        v_LUT[sbin*2 - 1 - i] = v_LUT[i] #weights for right-hand pixels 

    print v_LUT

#input: [[-1, 0], ... [4, 5]] where each sub-array i is the list of bins where pixel i contributes.
#output: [[-1 ... 6], [3 ... 7]] where each sub-array i is the list of pixels in bin i
def scatter_to_gather(px_to_ixp):
    ixp_max=20 #TODO: set this to # unique elements in sub-arrays of input.
    ixp_to_px = []
    for i in xrange(0, ixp_max): #preallocate list
        ixp_to_px.append([])

    for ixp in xrange(0, ixp_max): #TODO: auto-determine this    
        for px in xrange(0, len(px_to_ixp)):
            if ixp in px_to_ixp[px]:
                ixp_to_px[ixp].append(px)
                #print 'ixp=%d, px=%d' %(ixp, px)
                #print ('appending new data... ixp_to_px[%d] = ' %ixp), ixp_to_px[ixp] 
    return ixp_to_px

#quick glance at the data
def test_by_eyeball():
    print "\n***test_by_eyeball()"
    sbin=4
    print " scatter:"
    for x in xrange(0, 20):
        print x, ixp_scatter(x, sbin)

    print " gather:"
    for ixp in xrange(0, 10):
        print ixp, px_gather(ixp, sbin)

def gather_vs_scatter_correctness():
    print "\n***gather_vs_scatter_correctness()" 
    sbin = 4
    num_px = 50

    #ground truth (scatter)
    px_to_ixp_GT = [ixp_scatter(px, sbin) for px in xrange(0, num_px)]
    ixp_to_px_GT = scatter_to_gather(px_to_ixp_GT)

    print "ground truth computed from voc5 scatter code:"
    for px in xrange(0, len(ixp_to_px_GT)):
        print ("px_to_ixp_GT[%d] = " %px), ixp_to_px_GT[px]


    #experimental (gather)
    print "experimental gather code:"
    px_to_ixp_experimental = [px_gather(ixp, sbin) for ixp in xrange(0, 20)]
    for px in xrange(0, len(px_to_ixp_experimental)):
        print ("px_to_ixp_experimental[%d] = " %px), px_to_ixp_experimental[px] 

def test_scatter_gather_weights(sbin):
    print "\n***gather_vs_scatter_weights()"
    #sbin = 4
    num_px = 50

    #ground truth (scatter)
    px_to_vx0_GT = [px_to_vx0(px, sbin) for px in xrange(0, num_px)] 
    for px in xrange(0, len(px_to_vx0_GT)):
        print ("px_to_vx0_GT[%d] = " %px), px_to_vx0_GT[px]


if __name__ == "__main__":

    #test_by_eyeball()
    #gather_vs_scatter_correctness()

    sbin = 7
    test_scatter_gather_weights(sbin)

    #test_get_vx0(sbin)
    init_v_LUT(sbin)


