program derivetive_Lya_function
    implicit none

    double precision :: x_k1, x_k2, x_k3, x_k4
    double precision :: y_k1, y_k2, y_k3, y_k4
    double precision :: z_k1, z_k2, z_k3, z_k4
    double precision :: w_k1, w_k2, w_k3, w_k4
    double precision :: phi_k1, phi_k2, phi_k3, phi_k4
    double precision :: x_d, y_d, z_d, w_d, phi_d

    double precision :: x_r_k1, x_r_k2, x_r_k3, x_r_k4
    double precision :: y_r_k1, y_r_k2, y_r_k3, y_r_k4
    double precision :: z_r_k1, z_r_k2, z_r_k3, z_r_k4
    double precision :: phi_r_k1, phi_r_k2, phi_r_k3, phi_r_k4
    double precision :: x_r, y_s, z_s, phi_s

    double precision :: e_1_k1, e_1_k2, e_1_k3, e_1_k4
    double precision :: e_2_k1, e_2_k2, e_2_k3, e_2_k4
    double precision :: e_3_k1, e_3_k2, e_3_k3, e_3_k4
    double precision :: e_4_k1, e_4_k2, e_4_k3, e_4_k4
    double precision :: e_1, e_2, e_3, e_4

    double precision :: a_t_k1, a_t_k2, a_t_k3, a_t_k4
    double precision :: b_t_k1, b_t_k2, b_t_k3, b_t_k4
    double precision :: d_t_k1, d_t_k2, d_t_k3, d_t_k4
    double precision :: r_t_k1, r_t_k2, r_t_k3, r_t_k4
    double precision :: a_t, b_t, d_t, r_t

    double precision :: g_1_k1, g_1_k2, g_1_k3, g_1_k4
    double precision :: g_2_k1, g_2_k2, g_2_k3, g_2_k4
    double precision :: g_3_k1, g_3_k2, g_3_k3, g_3_k4
    double precision :: g_4_k1, g_4_k2, g_4_k3, g_4_k4
    double precision :: g_1, g_2, g_3, g_4

    double precision :: e_1_l, e_2_l, e_3_l, e_4_l
    double precision :: trans, t, max_lya_fxn, lya_fxn
    double precision :: k_1, k_2 ! main varying param

    double precision, parameter :: a = 3.0, b = 1.0, alpha = 0.1, beta = 0.02, I = 3.1, c = 1.0, d = 5.0
    double precision, parameter :: sigma = 0.0278, r = 0.006, s = 4.75
    double precision, parameter :: x_0 = -1.56, y_0 = -1.619, mu = 0.0009, gamma = 3.0, delta = 0.9573

    double precision, parameter :: l_1 = 1, l_2 = 1, l_3 = 1, l_4 = 1, threshold = 0.015

    double precision, parameter :: tmin = 0.0, tmax = 22000, h = 0.001

    OPEN(10, File = "Lya_k_1_k_2.txt", Status = 'unknown')

    trans = tmax - 5500

    k_1 = 0
    do while (k_1.LE.5)
        k_2 = 0
        do while (k_2.LE.2)

            x_d = 1.D0; y_d = 0.5D0 ;z_d = 1.3D0;w_d = -0.5D0;phi_d = -1.2D0
            x_r = 1.1D0; y_s = -2.2D0 ;z_s = -0.6D0;phi_s = 0.5D0
            e_1 = 2; e_2 = -2 ;e_3 = 2;e_4 = -2
            a_t = 0; b_t = 0; d_t = 0; r_t = 0
            g_1 = 0.5; g_2 = 0.5 ;g_3 = 0.5;g_4 = 0.5
            max_lya_fxn = -100000

        do t = tmin, tmax, h

            x_k1 = h * drive_system_x(x_d, y_d, z_d, w_d, phi_d)
            y_k1 = h * drive_system_y(x_d, y_d, z_d, w_d, phi_d)
            z_k1 = h * drive_system_z(x_d, y_d, z_d, w_d, phi_d)
            w_k1 = h * drive_system_w(x_d, y_d, z_d, w_d, phi_d)
            phi_k1 = h * drive_system_phi(x_d, y_d, z_d, w_d, phi_d)

            x_r_k1 = h * response_system_x(x_r, y_s, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2, &
                    g_3, e_3, g_4, e_4)
            y_r_k1 = h * response_system_y(x_r, y_s, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2, &
                    g_3, e_3, g_4, e_4)
            z_r_k1 = h * response_system_z(x_r, y_s, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2, &
                    g_3, e_3, g_4, e_4)
            phi_r_k1 = h * response_system_phi(x_r, y_s, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2, &
                    g_3, e_3, g_4, e_4)

            e_1_k1 = h * error_system_e_1(x_r, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2 &
                    , g_3, e_3, g_4, e_4, w_d)
            e_2_k1 = h * error_system_e_2(x_r, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2 &
                    , g_3, e_3, g_4, e_4, w_d)
            e_3_k1 = h * error_system_e_3(x_r, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2 &
                    , g_3, e_3, g_4, e_4, w_d)
            e_4_k1 = h * error_system_e_4(x_r, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2 &
                    , g_3, e_3, g_4, e_4, w_d)

            a_t_k1 = h * uncertainty_system_a_t(x_r, z_s, e_1, e_2, e_3)
            b_t_k1 = h * uncertainty_system_b_t(x_r, z_s, e_1, e_2, e_3)
            d_t_k1 = h * uncertainty_system_d_t(x_r, z_s, e_1, e_2, e_3)
            r_t_k1 = h * uncertainty_system_r_t(x_r, z_s, e_1, e_2, e_3)

            g_1_k1 = h * gain_system_g_1(e_1, e_2, e_3, e_4)
            g_2_k1 = h * gain_system_g_2(e_1, e_2, e_3, e_4)
            g_3_k1 = h * gain_system_g_3(e_1, e_2, e_3, e_4)
            g_4_k1 = h * gain_system_g_4(e_1, e_2, e_3, e_4)

            ! Calculate k2
            x_k2 = h * drive_system_x(x_d + x_k1 / 2, y_d + y_k1 / 2, z_d + z_k1 / 2, &
                    w_d + w_k1 / 2, phi_d + phi_k1 / 2)
            y_k2 = h * drive_system_y(x_d + x_k1 / 2, y_d + y_k1 / 2, z_d + z_k1 / 2, &
                    w_d + w_k1 / 2, phi_d + phi_k1 / 2)
            z_k2 = h * drive_system_z(x_d + x_k1 / 2, y_d + y_k1 / 2, z_d + z_k1 / 2, &
                    w_d + w_k1 / 2, phi_d + phi_k1 / 2)
            w_k2 = h * drive_system_w(x_d + x_k1 / 2, y_d + y_k1 / 2, z_d + z_k1 / 2, &
                    w_d + w_k1 / 2, phi_d + phi_k1 / 2)
            phi_k2 = h * drive_system_phi(x_d + x_k1 / 2, y_d + y_k1 / 2, z_d + z_k1 / 2, &
                    w_d + w_k1 / 2, phi_d + phi_k1 / 2)

            x_r_k2 = h * response_system_x(x_r + x_r_k1 / 2, y_s + y_r_k1 / 2, z_s + z_r_k1 / 2, &
                    phi_s + phi_r_k1 / 2, &
                    a_t + a_t_k1 / 2, b_t + b_t_k1 / 2, d_t + d_t_k1 / 2, r_t + r_t_k1 / 2, &
                    g_1 + g_1_k1 / 2, e_1 + e_1_k1 / 2, g_2 + g_2_k1 / 2, e_2 + e_2_k1 / 2, &
                    g_3 + g_3_k1 / 2, e_3 + e_3_k1 / 2, g_4 + g_4_k1 / 2, e_4 + e_4_k1 / 2)

            y_r_k2 = h * response_system_y(x_r + x_r_k1 / 2, y_s + y_r_k1 / 2, z_s + z_r_k1 / 2, &
                    phi_s + phi_r_k1 / 2, &
                    a_t + a_t_k1 / 2, b_t + b_t_k1 / 2, d_t + d_t_k1 / 2, r_t + r_t_k1 / 2, &
                    g_1 + g_1_k1 / 2, e_1 + e_1_k1 / 2, g_2 + g_2_k1 / 2, e_2 + e_2_k1 / 2, &
                    g_3 + g_3_k1 / 2, e_3 + e_3_k1 / 2, g_4 + g_4_k1 / 2, e_4 + e_4_k1 / 2)

            z_r_k2 = h * response_system_z(x_r + x_r_k1 / 2, y_s + y_r_k1 / 2, z_s + z_r_k1 / 2, &
                    phi_s + phi_r_k1 / 2, &
                    a_t + a_t_k1 / 2, b_t + b_t_k1 / 2, d_t + d_t_k1 / 2, r_t + r_t_k1 / 2, &
                    g_1 + g_1_k1 / 2, e_1 + e_1_k1 / 2, g_2 + g_2_k1 / 2, e_2 + e_2_k1 / 2, &
                    g_3 + g_3_k1 / 2, e_3 + e_3_k1 / 2, g_4 + g_4_k1 / 2, e_4 + e_4_k1 / 2)

            phi_r_k2 = h * response_system_phi(x_r + x_r_k1 / 2, y_s + y_r_k1 / 2, z_s + z_r_k1 / 2, &
                    phi_s + phi_r_k1 / 2, &
                    a_t + a_t_k1 / 2, b_t + b_t_k1 / 2, d_t + d_t_k1 / 2, r_t + r_t_k1 / 2, &
                    g_1 + g_1_k1 / 2, e_1 + e_1_k1 / 2, g_2 + g_2_k1 / 2, e_2 + e_2_k1 / 2, &
                    g_3 + g_3_k1 / 2, e_3 + e_3_k1 / 2, g_4 + g_4_k1 / 2, e_4 + e_4_k1 / 2)

            e_1_k2 = h * error_system_e_1(x_r + x_r_k1 / 2, z_s + z_r_k1 / 2, phi_s + phi_r_k1 / 2, &
                    a_t + a_t_k1 / 2, b_t + b_t_k1 / 2, d_t + d_t_k1 / 2, r_t + r_t_k1 / 2, &
                    g_1 + g_1_k1 / 2, e_1 + e_1_k1 / 2, g_2 + g_2_k1 / 2, e_2 + e_2_k1 / 2, &
                    g_3 + g_3_k1 / 2, e_3 + e_3_k1 / 2, g_4 + g_4_k1 / 2, e_4 + e_4_k1 / 2, w_d + w_k1 / 2)

            e_2_k2 = h * error_system_e_2(x_r + x_r_k1 / 2, z_s + z_r_k1 / 2, phi_s + phi_r_k1 / 2, &
                    a_t + a_t_k1 / 2, b_t + b_t_k1 / 2, d_t + d_t_k1 / 2, r_t + r_t_k1 / 2, &
                    g_1 + g_1_k1 / 2, e_1 + e_1_k1 / 2, g_2 + g_2_k1 / 2, e_2 + e_2_k1 / 2, &
                    g_3 + g_3_k1 / 2, e_3 + e_3_k1 / 2, g_4 + g_4_k1 / 2, e_4 + e_4_k1 / 2, &
                    w_d + w_k1 / 2)

            e_3_k2 = h * error_system_e_3(x_r + x_r_k1 / 2, z_s + z_r_k1 / 2, phi_s + phi_r_k1 / 2, &
                    a_t + a_t_k1 / 2, b_t + b_t_k1 / 2, d_t + d_t_k1 / 2, r_t + r_t_k1 / 2, &
                    g_1 + g_1_k1 / 2, e_1 + e_1_k1 / 2, g_2 + g_2_k1 / 2, e_2 + e_2_k1 / 2, &
                    g_3 + g_3_k1 / 2, e_3 + e_3_k1 / 2, g_4 + g_4_k1 / 2, e_4 + e_4_k1 / 2, &
                    w_d + w_k1 / 2)

            e_4_k2 = h * error_system_e_4(x_r + x_r_k1 / 2, z_s + z_r_k1 / 2, phi_s + phi_r_k1 / 2, &
                    a_t + a_t_k1 / 2, b_t + b_t_k1 / 2, d_t + d_t_k1 / 2, r_t + r_t_k1 / 2, &
                    g_1 + g_1_k1 / 2, e_1 + e_1_k1 / 2, g_2 + g_2_k1 / 2, e_2 + e_2_k1 / 2, &
                    g_3 + g_3_k1 / 2, e_3 + e_3_k1 / 2, g_4 + g_4_k1 / 2, e_4 + e_4_k1 / 2, &
                    w_d + w_k1 / 2)

            a_t_k2 = h * uncertainty_system_a_t(x_r + x_r_k1 / 2, z_s + z_r_k1 / 2, e_1 + e_1_k1 / 2, &
                    e_2 + e_2_k1 / 2, e_3 + e_3_k1 / 2)
            b_t_k2 = h * uncertainty_system_b_t(x_r + x_r_k1 / 2, z_s + z_r_k1 / 2, e_1 + e_1_k1 / 2, &
                    e_2 + e_2_k1 / 2, e_3 + e_3_k1 / 2)
            d_t_k2 = h * uncertainty_system_d_t(x_r + x_r_k1 / 2, z_s + z_r_k1 / 2, e_1 + e_1_k1 / 2, &
                    e_2 + e_2_k1 / 2, e_3 + e_3_k1 / 2)
            r_t_k2 = h * uncertainty_system_r_t(x_r + x_r_k1 / 2, z_s + z_r_k1 / 2, e_1 + e_1_k1 / 2, &
                    e_2 + e_2_k1 / 2, e_3 + e_3_k1 / 2)

            g_1_k2 = h * gain_system_g_1(e_1 + e_1_k1 / 2, e_2 + e_2_k1 / 2, e_3 + e_3_k1 / 2, e_4 + e_4_k1 / 2)
            g_2_k2 = h * gain_system_g_2(e_1 + e_1_k1 / 2, e_2 + e_2_k1 / 2, e_3 + e_3_k1 / 2, e_4 + e_4_k1 / 2)
            g_3_k2 = h * gain_system_g_3(e_1 + e_1_k1 / 2, e_2 + e_2_k1 / 2, e_3 + e_3_k1 / 2, e_4 + e_4_k1 / 2)
            g_4_k2 = h * gain_system_g_4(e_1 + e_1_k1 / 2, e_2 + e_2_k1 / 2, e_3 + e_3_k1 / 2, e_4 + e_4_k1 / 2)



            ! Calculate k3
            x_k3 = h * drive_system_x(x_d + x_k2 / 2, y_d + y_k2 / 2, z_d + z_k2 / 2, &
                    w_d + w_k2 / 2, phi_d + phi_k2 / 2)
            y_k3 = h * drive_system_y(x_d + x_k2 / 2, y_d + y_k2 / 2, z_d + z_k2 / 2, &
                    w_d + w_k2 / 2, phi_d + phi_k2 / 2)
            z_k3 = h * drive_system_z(x_d + x_k2 / 2, y_d + y_k2 / 2, z_d + z_k2 / 2, &
                    w_d + w_k2 / 2, phi_d + phi_k2 / 2)
            w_k3 = h * drive_system_w(x_d + x_k2 / 2, y_d + y_k2 / 2, z_d + z_k2 / 2, &
                    w_d + w_k2 / 2, phi_d + phi_k2 / 2)
            phi_k3 = h * drive_system_phi(x_d + x_k2 / 2, y_d + y_k2 / 2, z_d + z_k2 / 2, &
                    w_d + w_k2 / 2, phi_d + phi_k2 / 2)

            x_r_k3 = h * response_system_x(x_r + x_r_k2 / 2, y_s + y_r_k2 / 2, z_s + z_r_k2 / 2, &
                    phi_s + phi_r_k2 / 2, &
                    a_t + a_t_k2 / 2, b_t + b_t_k2 / 2, d_t + d_t_k2 / 2, r_t + r_t_k2 / 2, &
                    g_1 + g_1_k2 / 2, e_1 + e_1_k2 / 2, g_2 + g_2_k2 / 2, e_2 + e_2_k2 / 2, &
                    g_3 + g_3_k2 / 2, e_3 + e_3_k2 / 2, g_4 + g_4_k2 / 2, e_4 + e_4_k2 / 2)
            y_r_k3 = h * response_system_y(x_r + x_r_k2 / 2, y_s + y_r_k2 / 2, z_s + z_r_k2 / 2, &
                    phi_s + phi_r_k2 / 2, &
                    a_t + a_t_k2 / 2, b_t + b_t_k2 / 2, d_t + d_t_k2 / 2, r_t + r_t_k2 / 2, &
                    g_1 + g_1_k2 / 2, e_1 + e_1_k2 / 2, g_2 + g_2_k2 / 2, e_2 + e_2_k2 / 2, &
                    g_3 + g_3_k2 / 2, e_3 + e_3_k2 / 2, g_4 + g_4_k2 / 2, e_4 + e_4_k2 / 2)

            z_r_k3 = h * response_system_z(x_r + x_r_k2 / 2, y_s + y_r_k2 / 2, z_s + z_r_k2 / 2, &
                    phi_s + phi_r_k2 / 2, &
                    a_t + a_t_k2 / 2, b_t + b_t_k2 / 2, d_t + d_t_k2 / 2, r_t + r_t_k2 / 2, &
                    g_1 + g_1_k2 / 2, e_1 + e_1_k2 / 2, g_2 + g_2_k2 / 2, e_2 + e_2_k2 / 2, &
                    g_3 + g_3_k2 / 2, e_3 + e_3_k2 / 2, g_4 + g_4_k2 / 2, e_4 + e_4_k2 / 2)

            phi_r_k3 = h * response_system_phi(x_r + x_r_k2 / 2, y_s + y_r_k2 / 2, z_s + z_r_k2 / 2, &
                    phi_s + phi_r_k2 / 2, &
                    a_t + a_t_k2 / 2, b_t + b_t_k2 / 2, d_t + d_t_k2 / 2, r_t + r_t_k2 / 2, &
                    g_1 + g_1_k2 / 2, e_1 + e_1_k2 / 2, g_2 + g_2_k2 / 2, e_2 + e_2_k2 / 2, &
                    g_3 + g_3_k2 / 2, e_3 + e_3_k2 / 2, g_4 + g_4_k2 / 2, e_4 + e_4_k2 / 2)

            e_1_k3 = h * error_system_e_1(x_r + x_r_k2 / 2, z_s + z_r_k2 / 2, phi_s + phi_r_k2 / 2, &
                    a_t + a_t_k2 / 2, b_t + b_t_k2 / 2, d_t + d_t_k2 / 2, r_t + r_t_k2 / 2, &
                    g_1 + g_1_k2 / 2, e_1 + e_1_k2 / 2, g_2 + g_2_k2 / 2, e_2 + e_2_k2 / 2, &
                    g_3 + g_3_k2 / 2, e_3 + e_3_k2 / 2, g_4 + g_4_k2 / 2, e_4 + e_4_k2 / 2, w_d + w_k2 / 2)

            e_2_k3 = h * error_system_e_2(x_r + x_r_k2 / 2, z_s + z_r_k2 / 2, phi_s + phi_r_k2 / 2, &
                    a_t + a_t_k2 / 2, b_t + b_t_k2 / 2, d_t + d_t_k2 / 2, r_t + r_t_k2 / 2, &
                    g_1 + g_1_k2 / 2, e_1 + e_1_k2 / 2, g_2 + g_2_k2 / 2, e_2 + e_2_k2 / 2, &
                    g_3 + g_3_k2 / 2, e_3 + e_3_k2 / 2, g_4 + g_4_k2 / 2, e_4 + e_4_k2 / 2, &
                    w_d + w_k2 / 2)

            e_3_k3 = h * error_system_e_3(x_r + x_r_k2 / 2, z_s + z_r_k2 / 2, phi_s + phi_r_k2 / 2, &
                    a_t + a_t_k2 / 2, b_t + b_t_k2 / 2, d_t + d_t_k2 / 2, r_t + r_t_k2 / 2, &
                    g_1 + g_1_k2 / 2, e_1 + e_1_k2 / 2, g_2 + g_2_k2 / 2, e_2 + e_2_k2 / 2, &
                    g_3 + g_3_k2 / 2, e_3 + e_3_k2 / 2, g_4 + g_4_k2 / 2, e_4 + e_4_k2 / 2, &
                    w_d + w_k2 / 2)

            e_4_k3 = h * error_system_e_4(x_r + x_r_k2 / 2, z_s + z_r_k2 / 2, phi_s + phi_r_k2 / 2, &
                    a_t + a_t_k2 / 2, b_t + b_t_k2 / 2, d_t + d_t_k2 / 2, r_t + r_t_k2 / 2, &
                    g_1 + g_1_k2 / 2, e_1 + e_1_k2 / 2, g_2 + g_2_k2 / 2, e_2 + e_2_k2 / 2, &
                    g_3 + g_3_k2 / 2, e_3 + e_3_k2 / 2, g_4 + g_4_k2 / 2, e_4 + e_4_k2 / 2, &
                    w_d + w_k2 / 2)

            a_t_k3 = h * uncertainty_system_a_t(x_r + x_r_k2 / 2, z_s + z_r_k2 / 2, e_1 + e_1_k2 / 2, &
                    e_2 + e_2_k2 / 2, e_3 + e_3_k2 / 2)
            b_t_k3 = h * uncertainty_system_b_t(x_r + x_r_k2 / 2, z_s + z_r_k2 / 2, e_1 + e_1_k2 / 2, &
                    e_2 + e_2_k2 / 2, e_3 + e_3_k2 / 2)
            d_t_k3 = h * uncertainty_system_d_t(x_r + x_r_k2 / 2, z_s + z_r_k2 / 2, e_1 + e_1_k2 / 2, &
                    e_2 + e_2_k2 / 2, e_3 + e_3_k2 / 2)
            r_t_k3 = h * uncertainty_system_r_t(x_r + x_r_k2 / 2, z_s + z_r_k2 / 2, e_1 + e_1_k2 / 2, &
                    e_2 + e_2_k2 / 2, e_3 + e_3_k2 / 2)

            g_1_k3 = h * gain_system_g_1(e_1 + e_1_k2 / 2, e_2 + e_2_k2 / 2, e_3 + e_3_k2 / 2, e_4 + e_4_k2 / 2)
            g_2_k3 = h * gain_system_g_2(e_1 + e_1_k2 / 2, e_2 + e_2_k2 / 2, e_3 + e_3_k2 / 2, e_4 + e_4_k2 / 2)
            g_3_k3 = h * gain_system_g_3(e_1 + e_1_k2 / 2, e_2 + e_2_k2 / 2, e_3 + e_3_k2 / 2, e_4 + e_4_k2 / 2)
            g_4_k3 = h * gain_system_g_4(e_1 + e_1_k2 / 2, e_2 + e_2_k2 / 2, e_3 + e_3_k2 / 2, e_4 + e_4_k2 / 2)


            ! Calculate k4
            x_k4 = h * drive_system_x(x_d + x_k3, y_d + y_k3, z_d + z_k3, &
                    w_d + w_k3, phi_d + phi_k3)
            y_k4 = h * drive_system_y(x_d + x_k3, y_d + y_k3, z_d + z_k3, &
                    w_d + w_k3, phi_d + phi_k3)
            z_k4 = h * drive_system_z(x_d + x_k3, y_d + y_k3, z_d + z_k3, &
                    w_d + w_k3, phi_d + phi_k3)
            w_k4 = h * drive_system_w(x_d + x_k3, y_d + y_k3, z_d + z_k3, &
                    w_d + w_k3, phi_d + phi_k3)
            phi_k4 = h * drive_system_phi(x_d + x_k3, y_d + y_k3, z_d + z_k3, &
                    w_d + w_k3, phi_d + phi_k3)

            x_r_k4 = h * response_system_x(x_r + x_r_k3, y_s + y_r_k3, z_s + z_r_k3, &
                    phi_s + phi_r_k3, &
                    a_t + a_t_k3, b_t + b_t_k3, d_t + d_t_k3, r_t + r_t_k3, &
                    g_1 + g_1_k3, e_1 + e_1_k3, g_2 + g_2_k3, e_2 + e_2_k3, &
                    g_3 + g_3_k3, e_3 + e_3_k3, g_4 + g_4_k3, e_4 + e_4_k3)

            y_r_k4 = h * response_system_y(x_r + x_r_k3, y_s + y_r_k3, z_s + z_r_k3, &
                    phi_s + phi_r_k3, &
                    a_t + a_t_k3, b_t + b_t_k3, d_t + d_t_k3, r_t + r_t_k3, &
                    g_1 + g_1_k3, e_1 + e_1_k3, g_2 + g_2_k3, e_2 + e_2_k3, &
                    g_3 + g_3_k3, e_3 + e_3_k3, g_4 + g_4_k3, e_4 + e_4_k3)

            z_r_k4 = h * response_system_z(x_r + x_r_k3, y_s + y_r_k3, z_s + z_r_k3, &
                    phi_s + phi_r_k3, &
                    a_t + a_t_k3, b_t + b_t_k3, d_t + d_t_k3, r_t + r_t_k3, &
                    g_1 + g_1_k3, e_1 + e_1_k3, g_2 + g_2_k3, e_2 + e_2_k3, &
                    g_3 + g_3_k3, e_3 + e_3_k3, g_4 + g_4_k3, e_4 + e_4_k3)

            phi_r_k4 = h * response_system_phi(x_r + x_r_k3, y_s + y_r_k3, z_s + z_r_k3, &
                    phi_s + phi_r_k3, &
                    a_t + a_t_k3, b_t + b_t_k3, d_t + d_t_k3, r_t + r_t_k3, &
                    g_1 + g_1_k3, e_1 + e_1_k3, g_2 + g_2_k3, e_2 + e_2_k3, &
                    g_3 + g_3_k3, e_3 + e_3_k3, g_4 + g_4_k3, e_4 + e_4_k3)

            e_1_k4 = h * error_system_e_1(x_r + x_r_k3, z_s + z_r_k3, phi_s + phi_r_k3, &
                    a_t + a_t_k3, b_t + b_t_k3, d_t + d_t_k3, r_t + r_t_k3, &
                    g_1 + g_1_k3, e_1 + e_1_k3, g_2 + g_2_k3, e_2 + e_2_k3, &
                    g_3 + g_3_k3, e_3 + e_3_k3, g_4 + g_4_k3, e_4 + e_4_k3, w_d + w_k3)

            e_2_k4 = h * error_system_e_2(x_r + x_r_k3, z_s + z_r_k3, phi_s + phi_r_k3, &
                    a_t + a_t_k3, b_t + b_t_k3, d_t + d_t_k3, r_t + r_t_k3, &
                    g_1 + g_1_k3, e_1 + e_1_k3, g_2 + g_2_k3, e_2 + e_2_k3, &
                    g_3 + g_3_k3, e_3 + e_3_k3, g_4 + g_4_k3, e_4 + e_4_k3, &
                    w_d + w_k3)

            e_3_k4 = h * error_system_e_3(x_r + x_r_k3, z_s + z_r_k3, phi_s + phi_r_k3, &
                    a_t + a_t_k3, b_t + b_t_k3, d_t + d_t_k3, r_t + r_t_k3, &
                    g_1 + g_1_k3, e_1 + e_1_k3, g_2 + g_2_k3, e_2 + e_2_k3, &
                    g_3 + g_3_k3, e_3 + e_3_k3, g_4 + g_4_k3, e_4 + e_4_k3, &
                    w_d + w_k3)

            e_4_k4 = h * error_system_e_4(x_r + x_r_k3, z_s + z_r_k3, phi_s + phi_r_k3, &
                    a_t + a_t_k3, b_t + b_t_k3, d_t + d_t_k3, r_t + r_t_k3, &
                    g_1 + g_1_k3, e_1 + e_1_k3, g_2 + g_2_k3, e_2 + e_2_k3, &
                    g_3 + g_3_k3, e_3 + e_3_k3, g_4 + g_4_k3, e_4 + e_4_k3, &
                    w_d + w_k3)

            a_t_k4 = h * uncertainty_system_a_t(x_r + x_r_k3, z_s + z_r_k3, e_1 + e_1_k3, &
                    e_2 + e_2_k3, e_3 + e_3_k3)
            b_t_k4 = h * uncertainty_system_b_t(x_r + x_r_k3, z_s + z_r_k3, e_1 + e_1_k3, &
                    e_2 + e_2_k3, e_3 + e_3_k3)
            d_t_k4 = h * uncertainty_system_d_t(x_r + x_r_k3, z_s + z_r_k3, e_1 + e_1_k3, &
                    e_2 + e_2_k3, e_3 + e_3_k3)
            r_t_k4 = h * uncertainty_system_r_t(x_r + x_r_k3, z_s + z_r_k3, e_1 + e_1_k3, &
                    e_2 + e_2_k3, e_3 + e_3_k3)

            g_1_k4 = h * gain_system_g_1(e_1 + e_1_k3, e_2 + e_2_k3, e_3 + e_3_k3, e_4 + e_4_k3)
            g_2_k4 = h * gain_system_g_2(e_1 + e_1_k3, e_2 + e_2_k3, e_3 + e_3_k3, e_4 + e_4_k3)
            g_3_k4 = h * gain_system_g_3(e_1 + e_1_k3, e_2 + e_2_k3, e_3 + e_3_k3, e_4 + e_4_k3)
            g_4_k4 = h * gain_system_g_4(e_1 + e_1_k3, e_2 + e_2_k3, e_3 + e_3_k3, e_4 + e_4_k3)



            ! param update
            x_d = x_d + (x_k1 + 2 * x_k2 + 2 * x_k3 + x_k4) / 6
            y_d = y_d + (y_k1 + 2 * y_k2 + 2 * y_k3 + y_k4) / 6
            z_d = z_d + (z_k1 + 2 * z_k2 + 2 * z_k3 + z_k4) / 6
            w_d = w_d + (w_k1 + 2 * w_k2 + 2 * w_k3 + w_k4) / 6
            phi_d = phi_d + (phi_k1 + 2 * phi_k2 + 2 * phi_k3 + phi_k4) / 6

            x_r = x_r + (x_r_k1 + 2 * x_r_k2 + 2 * x_r_k3 + x_r_k4) / 6
            y_s = y_s + (y_r_k1 + 2 * y_r_k2 + 2 * y_r_k3 + y_r_k4) / 6
            z_s = z_s + (z_r_k1 + 2 * z_r_k2 + 2 * z_r_k3 + z_r_k4) / 6
            phi_s = phi_s + (phi_r_k1 + 2 * phi_r_k2 + 2 * phi_r_k3 + phi_r_k4) / 6

            e_1 = e_1 + (e_1_k1 + 2 * e_1_k2 + 2 * e_1_k3 + e_1_k4) / 6
            e_2 = e_2 + (e_2_k1 + 2 * e_2_k2 + 2 * e_2_k3 + e_2_k4) / 6
            e_3 = e_3 + (e_3_k1 + 2 * e_3_k2 + 2 * e_3_k3 + e_3_k4) / 6
            e_4 = e_4 + (e_4_k1 + 2 * e_4_k2 + 2 * e_4_k3 + e_4_k4) / 6

            a_t = a_t + (a_t_k1 + 2 * a_t_k2 + 2 * a_t_k3 + a_t_k4) / 6
            b_t = b_t + (b_t_k1 + 2 * b_t_k2 + 2 * b_t_k3 + b_t_k4) / 6
            d_t = d_t + (d_t_k1 + 2 * d_t_k2 + 2 * d_t_k3 + d_t_k4) / 6
            r_t = r_t + (r_t_k1 + 2 * r_t_k2 + 2 * r_t_k3 + r_t_k4) / 6

            g_1 = g_1 + (g_1_k1 + 2 * g_1_k2 + 2 * g_1_k3 + g_1_k4) / 6
            g_2 = g_2 + (g_2_k1 + 2 * g_2_k2 + 2 * g_2_k3 + g_2_k4) / 6
            g_3 = g_3 + (g_3_k1 + 2 * g_3_k2 + 2 * g_3_k3 + g_3_k4) / 6
            g_4 = g_4 + (g_4_k1 + 2 * g_4_k2 + 2 * g_4_k3 + g_4_k4) / 6

            if (ABS(e_1).LE.threshold.and.ABS(e_2).LE.threshold.and.ABS(e_3).LE.threshold.and.&
                    ABS(e_4).LE.threshold)then
                e_1_l = 0
                e_2_l = 0
                e_3_l = 0
                e_4_l = 0
            ELSE
                e_1_l = e_1
                e_2_l = e_2
                e_3_l = e_3
                e_4_l = e_4
            end if

            lya_fxn = (2 * a * x_r - 3 * b * x_r**2 - k_1 * alpha - 3 * k_1 * beta * phi_s **2 - l_1) * e_1_l**2 &
                    - (1 + l_2) * e_2_l**2 &
                    - (r + l_3) * e_3_l**2&
                    - (k_2 + l_4) * e_4_l**2&
                    + (1 - 2 * d * x_r) * e_1_l * e_2_l &
                    + (r*s -1) * e_1_l * e_3_l &
                    + (1 - 6 * k_1 * beta * x_r * phi_s) * e_1_l * e_4_l&
                    + sigma * w_d * e_2_l

            if (t.ge.trans.and.lya_fxn.ge.max_lya_fxn)then

                max_lya_fxn = lya_fxn

            end if


        end do

            write (10, *)k_1, k_2,max_lya_fxn

            k_2 = k_2 + 0.01
        end do

        k_1 = k_1 + 0.01
    end do
contains

    double precision function drive_system_x(x_d, y_d, z_d, w_d, phi_d)
        implicit none
        double precision, intent(in) :: x_d, y_d, z_d, w_d, phi_d
        drive_system_x = (a * x_d**2) - (b * x_d**3) + y_d - z_d - k_1 * (alpha + 3 * beta * phi_d**2) * x_d + I
    end function drive_system_x

    double precision function drive_system_y(x_d, y_d, z_d, w_d, phi_d)
        implicit none
        double precision, intent(in) :: x_d, y_d, z_d, w_d, phi_d
        drive_system_y = c - d * x_d ** 2 - y_d - sigma * w_d
    end function drive_system_y

    double precision function drive_system_z(x_d, y_d, z_d, w_d, phi_d)
        implicit none
        double precision, intent(in) :: x_d, y_d, z_d, w_d, phi_d
        drive_system_z = r * (s * (x_d - x_0) - z_d)
    end function drive_system_z

    double precision function drive_system_w(x_d, y_d, z_d, w_d, phi_d)
        implicit none
        double precision, intent(in) :: x_d, y_d, z_d, w_d, phi_d
        drive_system_w = mu * (gamma * (y_d - y_0) - delta * w_d)
    end function drive_system_w

    double precision function drive_system_phi(x_d, y_d, z_d, w_d, phi_d)
        implicit none
        double precision, intent(in) :: x_d, y_d, z_d, w_d, phi_d
        drive_system_phi = x_d - k_2 * phi_d
    end function drive_system_phi


    double precision function response_system_x(x_r, y_s, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2, &
            g_3, e_3, g_4, e_4)
        implicit none
        double precision, intent(in) :: x_r, y_s, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2, g_3, &
                e_3, g_4, e_4
        response_system_x = (a_t * x_r ** 2) - (b_t * x_r ** 3) + (y_s) - z_s - k_1 * (alpha + 3 * beta * phi_s ** 2)&
                * x_r + I - g_1 * e_1

    end function response_system_x

    double precision function response_system_y(x_r, y_s, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2, &
            g_3, e_3, g_4, e_4)
        implicit none
        double precision, intent(in) :: x_r, y_s, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2, g_3, &
                e_3, g_4, e_4
        response_system_y = c - d_t * x_r ** 2 - y_s - g_2 * e_2
    end function response_system_y

    double precision function response_system_z(x_r, y_s, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2, &
            g_3, e_3, g_4, e_4)
        implicit none
        double precision, intent(in) :: x_r, y_s, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2, g_3, &
                e_3, g_4, e_4
        response_system_z = r_t * (s * (x_r - x_0) - z_s) - g_3 * e_3
    end function response_system_z

    double precision function response_system_phi(x_r, y_s, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2, &
            g_3, e_3, g_4, e_4)
        implicit none
        double precision, intent(in) :: x_r, y_s, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2, g_3, &
                e_3, g_4, e_4
        response_system_phi = x_r - k_2 * phi_s - g_4 * e_4
    end function response_system_phi

    double precision function error_system_e_1(x_r, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2&
            , g_3, e_3, g_4, e_4, w_d)
        implicit none
        double precision, intent(in) :: x_r, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2&
                , g_3, e_3, g_4, e_4,  w_d
        error_system_e_1 = (2 * a * x_r - 3 * b * x_r ** 2 - k_1 * alpha - 3 * k_1 * beta * phi_s**2) * e_1 + e_2&
                - e_3 - 6 * k_1 * beta * x_r * phi_s * e_4 + (a_t - a) * x_r ** 2 - (b_t - b) * x_r ** 3&
                - g_1 * e_1
    end function error_system_e_1

    double precision function error_system_e_2(x_r, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2&
            , g_3, e_3, g_4, e_4, w_d)
        implicit none
        double precision, intent(in) :: x_r, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2&
                , g_3, e_3, g_4, e_4, w_d
        error_system_e_2 = -2 * d * x_r * e_1 - e_2 - (d_t - d) * x_r ** 2 + sigma * w_d - g_2 * e_2
    end function error_system_e_2

    double precision function error_system_e_3(x_r, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2&
            , g_3, e_3, g_4, e_4, w_d)
        implicit none
        double precision, intent(in) :: x_r, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2&
                , g_3, e_3, g_4, e_4, w_d
        error_system_e_3 = r * s * e_1 - r * e_3 + (r_t - r) * (s * (x_r - x_0) - z_s) - g_3 * e_3
    end function error_system_e_3

    double precision function error_system_e_4(x_r, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2&
            , g_3, e_3, g_4, e_4, w_d)
        implicit none
        double precision, intent(in) :: x_r, z_s, phi_s, a_t, b_t, d_t, r_t, g_1, e_1, g_2, e_2&
                , g_3, e_3, g_4, e_4, w_d
        error_system_e_4 = e_1 - k_2 * e_4 - g_4 * e_4
    end function error_system_e_4

    double precision function gain_system_g_1(e_1, e_2, e_3, e_4)
        implicit none
        double precision, intent(in) :: e_1, e_2, e_3, e_4
        gain_system_g_1 = e_1 ** 2
    end function gain_system_g_1

    double precision function gain_system_g_2(e_1, e_2, e_3, e_4)
        implicit none
        double precision, intent(in) :: e_1, e_2, e_3, e_4
        gain_system_g_2 = e_2 ** 2
    end function gain_system_g_2

    double precision function gain_system_g_3(e_1, e_2, e_3, e_4)
        implicit none
        double precision, intent(in) :: e_1, e_2, e_3, e_4
        gain_system_g_3 = e_3 ** 2
    end function gain_system_g_3

    double precision function gain_system_g_4(e_1, e_2, e_3, e_4)
        implicit none
        double precision, intent(in) :: e_1, e_2, e_3, e_4
        gain_system_g_4 = e_4 ** 2
    end function gain_system_g_4

    double precision function uncertainty_system_a_t(x_r, z_r, e_1, e_2, e_3)
        implicit none
        double precision, intent(in) :: x_r, z_r, e_1, e_2, e_3
        uncertainty_system_a_t = -x_r ** 2 * e_1
    end function uncertainty_system_a_t

    double precision function uncertainty_system_b_t(x_r, z_r, e_1, e_2, e_3)
        implicit none
        double precision, intent(in) :: x_r, z_r, e_1, e_2, e_3
        uncertainty_system_b_t = x_r ** 3 * e_1
    end function uncertainty_system_b_t

    double precision function uncertainty_system_d_t(x_r, z_r, e_1, e_2, e_3)
        implicit none
        double precision, intent(in) :: x_r, z_r, e_1, e_2, e_3
        uncertainty_system_d_t = x_r ** 2 * e_2
    end function uncertainty_system_d_t

    double precision function uncertainty_system_r_t(x_r, z_s, e_1, e_2, e_3)
        implicit none
        double precision, intent(in) :: x_r, z_s, e_1, e_2, e_3
        uncertainty_system_r_t = -(s * (x_r - x_0) - z_s) * e_3
    end function uncertainty_system_r_t


end program derivetive_Lya_function