# not many other energetic particles around the photon
def cut_isolation_et(photon_etcone20):
    return (photon_etcone20[:, 0] < 4000) & (photon_etcone20[:, 1] < 4000)

# narrow cone/spike
def cut_photon_reconstruction(photon_isTightID):
    return photon_isTightID[:, 0] & photon_isTightID[:, 1]