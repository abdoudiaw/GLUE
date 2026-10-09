/*
 * nn_interface.c — C socket client for B2.5-GNN coupling.
 *
 * Called from Fortran via ISO_C_BINDING. Connects to the SOLPEx
 * Python server over a Unix domain socket, sends plasma state,
 * receives volumetric source terms.
 *
 * Compile:
 *   gcc -c -O2 nn_interface.c -o nn_interface.o
 *
 * Link with B2.5 Fortran:
 *   ... nn_interface.o ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <stdint.h>

/* Persistent connection — opened once, reused across calls */
static int sock_fd = -1;
static const char *default_socket_path = "/tmp/solpex.sock";

/* ------------------------------------------------------------------ */
/* Internal helpers                                                    */
/* ------------------------------------------------------------------ */

static int ensure_connected(const char *path) {
    if (sock_fd >= 0) return 0;

    sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        perror("[nn_interface] socket");
        return -1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    if (connect(sock_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("[nn_interface] connect");
        close(sock_fd);
        sock_fd = -1;
        return -1;
    }

    fprintf(stderr, "[nn_interface] Connected to %s\n", path);
    return 0;
}

static int send_all(int fd, const void *buf, size_t len) {
    const char *p = (const char *)buf;
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, p + sent, len - sent, 0);
        if (n <= 0) {
            perror("[nn_interface] send");
            return -1;
        }
        sent += n;
    }
    return 0;
}

static int recv_all(int fd, void *buf, size_t len) {
    char *p = (char *)buf;
    size_t got = 0;
    while (got < len) {
        ssize_t n = recv(fd, p + got, len - got, 0);
        if (n <= 0) {
            perror("[nn_interface] recv");
            return -1;
        }
        got += n;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Public API — called from Fortran                                    */
/* ------------------------------------------------------------------ */

/*
 * nn_predict_sources:
 *   Sends plasma (Te, Ti, ne, ni, ua) to SOLPEx server.
 *   Receives volumetric sources (Sp, Qe, Qi, Sm).
 *
 * All arrays are Fortran column-major, double precision.
 * Dimensions: (-1:nx, -1:ny) but we only send the interior (0:nx-1, 0:ny-1).
 *
 * Called from Fortran as:
 *   call nn_predict_sources(nx, ny, ns, Te, Ti, ne_arr, ni_arr, ua_arr,
 *                           Sp, Qe, Qi, Sm, socket_path, ierr)
 */
void nn_predict_sources_(
    const int32_t *nx_p, const int32_t *ny_p, const int32_t *ns_p,
    const double *Te,     /* (-1:nx, -1:ny) */
    const double *Ti,     /* (-1:nx, -1:ny) */
    const double *ne_arr, /* (-1:nx, -1:ny) */
    const double *ni_arr, /* (-1:nx, -1:ny) */
    const double *ua_arr, /* (-1:nx, -1:ny) */
    double *Sp,           /* (-1:nx, -1:ny) output */
    double *Qe,           /* (-1:nx, -1:ny) output */
    double *Qi,           /* (-1:nx, -1:ny) output */
    double *Sm,           /* (-1:nx, -1:ny) output */
    const char *socket_path, /* Fortran string */
    int32_t *ierr,
    int socket_path_len   /* hidden Fortran string length */
) {
    int32_t nx = *nx_p;
    int32_t ny = *ny_p;
    int32_t ns = *ns_p;
    *ierr = 0;

    /* Parse socket path (trim Fortran trailing spaces) */
    char path[256];
    int plen = socket_path_len;
    while (plen > 0 && socket_path[plen-1] == ' ') plen--;
    if (plen <= 0 || plen >= 256) {
        strncpy(path, default_socket_path, sizeof(path));
    } else {
        memcpy(path, socket_path, plen);
        path[plen] = '\0';
    }

    /* Connect (or reuse existing connection) */
    if (ensure_connected(path) < 0) {
        *ierr = 1;
        return;
    }

    /*
     * Fortran arrays are (-1:nx, -1:ny) = (nx+2) x (ny+2).
     * B2.5 convention: interior is (0:nx-1, 0:ny-1).
     * We extract the interior and send as contiguous (ny, nx) blocks.
     */
    int ld = nx + 2;  /* leading dimension: -1..nx = nx+2 elements */
    int n_interior = ny * nx;
    double *sendbuf = (double *)malloc(5 * n_interior * sizeof(double));
    if (!sendbuf) {
        fprintf(stderr, "[nn_interface] malloc failed\n");
        *ierr = 2;
        return;
    }

    /* Pack 5 fields: extract interior (ix=0..nx-1, iy=0..ny-1) */
    const double *fields[5] = {Te, Ti, ne_arr, ni_arr, ua_arr};
    for (int f = 0; f < 5; f++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                /* Fortran (-1:nx, -1:ny) -> C index: (ix+1) + (iy+1)*ld */
                int fi = (ix + 1) + (iy + 1) * ld;
                sendbuf[f * n_interior + iy * nx + ix] = fields[f][fi];
            }
        }
    }

    /* Send header */
    int32_t header[3] = {nx, ny, ns};
    if (send_all(sock_fd, header, 12) < 0) { *ierr = 3; goto cleanup; }

    /* Send plasma */
    if (send_all(sock_fd, sendbuf, 5 * n_interior * sizeof(double)) < 0) {
        *ierr = 3; goto cleanup;
    }

    /* Receive sources: 4 * ny * nx float64 */
    double *recvbuf = (double *)malloc(4 * n_interior * sizeof(double));
    if (!recvbuf) { *ierr = 2; goto cleanup; }

    if (recv_all(sock_fd, recvbuf, 4 * n_interior * sizeof(double)) < 0) {
        *ierr = 4;
        free(recvbuf);
        goto cleanup;
    }

    /* Unpack into Fortran arrays (with guard cells zeroed) */
    double *outputs[4] = {Sp, Qe, Qi, Sm};
    for (int f = 0; f < 4; f++) {
        /* Zero full array including guard cells */
        memset(outputs[f], 0, (nx + 2) * (ny + 2) * sizeof(double));
        /* Fill interior */
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                int fi = (ix + 1) + (iy + 1) * ld;
                outputs[f][fi] = recvbuf[f * n_interior + iy * nx + ix];
            }
        }
    }
    free(recvbuf);

cleanup:
    free(sendbuf);

    if (*ierr != 0) {
        /* Close broken connection so next call reconnects */
        close(sock_fd);
        sock_fd = -1;
    }
}

/*
 * nn_disconnect: cleanly close the socket connection.
 * Call at B2.5 shutdown.
 */
void nn_disconnect_(void) {
    if (sock_fd >= 0) {
        close(sock_fd);
        sock_fd = -1;
        fprintf(stderr, "[nn_interface] Disconnected\n");
    }
}
