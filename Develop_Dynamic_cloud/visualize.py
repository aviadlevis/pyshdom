def viz3D(cloud_field):
    nx, ny, nz = cloud_field.shape
    dx, dy, dz = (1, 1, 1)

    xgrid = np.linspace(0, nx - 1, nx)
    ygrid = np.linspace(0, ny - 1, ny)
    zgrid = np.linspace(0, nz - 1, nz)
    X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
    figh = mlab.gcf()
    src = mlab.pipeline.scalar_field(X, Y, Z, cloud_field)
    src.spacing = [dx, dy, dz]
    src.update_image_data = True

    isosurface = mlab.pipeline.iso_surface(src, contours=[0.1 * cloud_field.max(), \
                                                          0.2 * cloud_field.max(), \
                                                          0.3 * cloud_field.max(), \
                                                          0.4 * cloud_field.max(), \
                                                          0.5 * cloud_field.max(), \
                                                          0.6 * cloud_field.max(), \
                                                          0.7 * cloud_field.max(), \
                                                          0.8 * cloud_field.max(), \
                                                          0.9 * cloud_field.max(), \
                                                          ], opacity=0.9)
    mlab.pipeline.volume(isosurface, figure=figh)
    color_bar = mlab.colorbar(title="volume", orientation='vertical', nb_labels=5)

    mlab.outline(figure=figh, color=(1, 1, 1))  # box around data axes
    mlab.orientation_axes(figure=figh)
    mlab.axes(figure=figh, xlabel="x (km)", ylabel="y (km)", zlabel="z (km)")
    mlab.show()