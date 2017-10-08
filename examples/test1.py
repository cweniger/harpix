def _test():
    dims = (10,3)
    D = np.ones(dims)
    h1 = Harpix(dims = dims).add_iso(32).add_random()
    h2 = Harpix(dims = dims).add_iso(64).add_random()

    h1.add_singularity((0,0), 0.1, 20, n = 100)#.add_random()
    h1.add_singularity((0,-10), 0.1, 20, n = 100)#.add_random()
    h2.add_singularity((0,10), 0.1, 20, n = 100)#.add_random()
    h1.add_func(lambda d: D*0.1/d, center = (0,10.0), mode='dist')
    h1.add_func(lambda d: D*0.1/d, center = (0,0.0), mode='dist')
    h2.add_func(lambda d: D*0.1/d, center = (0,-10.0), mode='dist')

    h1 += h2
    quit()

    T = get_trans_matrix(h1, h2)
    h2.data = _trans_data(T, h1.data)

    #print h2.get_integral()
#    T1 = get_trans_matrix(h1, 64)
#    T2 = get_trans_matrix(64, h1)
#    print h1.get_integral()

#    h1.data = T2.dot(T1.dot(h1.data))

    m = h1.get_healpix(256)
    hp.mollview(np.log10(m), nest =True)
    plt.savefig('test.eps')
    quit()
    #h2.add_random()
    T = get_trans_matrix(h1, h2)
    h2.data += T.dot(h1.data)
    print h2.get_integral()
    m = h2.get_healpix(256)
    hp.mollview(np.log10(m), nest =True)
    plt.savefig('test.eps')

    quit()

    h.add_iso(nside = 1)
    h.add_singularity((0,0.0), 0.1, 100, n = 100)#.add_random()
    h.add_func(lambda d: 0.1/d, center = (0,0.0), mode='dist')
    h.add_singularity((80,20), 0.1, 100, n = 100)#.add_random()
    #h.add_singularity((30,20), 0.1, 100, n = 1000)#.add_random()
    #h.add_func(lambda d: 0.1/d, center = (30,20), mode='dist')
    #h.add_random()
    #h.apply_mask(lambda l, b: abs(b) < 3)
    h.print_info()

    m = h.get_healpix(32)
    hp.mollview(np.log10(m), nest = True, cmap='gnuplot')
    #hp.cartview(np.log10(m), nest = True, cmap='gnuplot', lonra = [-1, 1],
          #  latra = [-1, 1])
    print h.get_integral()
    h.add_iso(nside = 4)
    h.add_singularity((0,0), 0.1, 100, n = 1000)#.add_random()
    h.add_func(lambda d: 0.1/d, center = (0,0), mode='dist')
    h.add_singularity((30,20), 0.1, 100, n = 1000)#.add_random()
    h.add_func(lambda d: 0.1/d, center = (30,20), mode='dist')
    h.add_random()
    #h.apply_mask(lambda l, b: abs(b) < 3)
    h.print_info()

    m = h.get_healpix(32)
    m = hp.smoothing(m, sigma =1, nest=True)
    hp.mollview(np.log10(m), nest = True, cmap='gnuplot')
    quit()
    quit()

    npix = hp.nside2npix(8)
    m = np.random.random((npix, 2,3))
    h=Harpix.from_healpix(m)
    m = h.get_healpix(128, idxs = (1,1))
    h.print_info()
    hp.mollview(np.log10(m), nest = True, cmap='gnuplot')
    plt.savefig('test2.eps')

    h = Harpix(dims=(10,)).add_iso(fill = 100)
    for i in range(10):
        lonlat = (40*i, 10*i)
        h0 = Harpix(dims=(10,))
        h0.add_peak(lonlat, .01, 10)
        print np.shape(h0.data)
        x = np.linspace(1, 10, 10)
        h0.add_func(lambda dist: x/(dist+0.01), mode = 'dist', center = lonlat)
        h += h0
    m = h.get_healpix(128, idxs=(4,))
    h.print_info()
    #hp.mollview(np.log10(m), nest = True, cmap='gnuplot', min = 1, max = 4)
    hp.cartview(np.log10(m), cmap='gnuplot', min = 1, max = 4)
    plt.savefig('test.eps')
    hp.mollview(np.log10(m), nest = True, cmap='gnuplot', min = 1, max = 4)
    plt.savefig('test3.eps')

if __name__ == "__main__":
    _test()
