c=======================================================================
c     b2mod_eirene_nn.F — Neural network replacement for EIRENE
c
c     Drop-in replacement for eirene_mc() that calls the SOLPEx GNN
c     server via Unix socket instead of running EIRENE Monte Carlo.
c
c     Usage in b2mod_eirene.F:
c       Replace the eirene_eirsrt() call with:
c
c       if (use_nn_sources) then
c         call eirene_mc_nn(nx, ny, ns, sna0, smo0, she0, shi0)
c       else
c         call eirene_eirsrt(...)
c       endif
c
c     The socket server must be running before B2.5 starts:
c       python coupler/solpex_b2_server.py --model eirene_gnn.pt \
c              --mesh coupling_dataset.npz --socket /tmp/solpex.sock
c
c     New input parameter in b2mn.dat:
c       b2mndr_eirene_nn  0    ! 0=EIRENE (default), 1=NN
c
c     Compilation:
c       Add to B2.5 object list. Link with nn_interface.o (C).
c=======================================================================

      module b2mod_eirene_nn_mod

      use b2mod_types
      implicit none

      integer, save :: use_nn = 0
      character(len=256), save :: nn_socket_path = '/tmp/solpex.sock'

      contains

c-----------------------------------------------------------------------
c     eirene_mc_nn: pack plasma -> call NN -> unpack sources
c-----------------------------------------------------------------------
      subroutine eirene_mc_nn(nx, ny, ns,
     &                        teb, tib, na, ua_in,
     &                        sna0, smo0, she0, shi0)

      use b2mod_types
      implicit none

      integer, intent(in) :: nx, ny, ns
      real(kind=R8), intent(in) :: teb(-1:nx,-1:ny)
      real(kind=R8), intent(in) :: tib(-1:nx,-1:ny)
      real(kind=R8), intent(in) :: na(-1:nx,-1:ny,0:1,0:ns-1)
      real(kind=R8), intent(in) :: ua_in(-1:nx,-1:ny,0:1,0:ns-1)

c     Output source arrays (same layout as EIRENE fills)
      real(kind=R8), intent(inout) :: sna0(-1:nx,-1:ny,0:1,0:ns-1)
      real(kind=R8), intent(inout) :: smo0(-1:nx,-1:ny,0:3,0:ns-1)
      real(kind=R8), intent(inout) :: she0(-1:nx,-1:ny,0:3)
      real(kind=R8), intent(inout) :: shi0(-1:nx,-1:ny,0:3)

c     Local arrays for NN interface (interior only)
      real(kind=R8) :: Te_2d(-1:nx,-1:ny)
      real(kind=R8) :: Ti_2d(-1:nx,-1:ny)
      real(kind=R8) :: ne_2d(-1:nx,-1:ny)
      real(kind=R8) :: ni_2d(-1:nx,-1:ny)
      real(kind=R8) :: ua_2d(-1:nx,-1:ny)
      real(kind=R8) :: Sp_2d(-1:nx,-1:ny)
      real(kind=R8) :: Qe_2d(-1:nx,-1:ny)
      real(kind=R8) :: Qi_2d(-1:nx,-1:ny)
      real(kind=R8) :: Sm_2d(-1:nx,-1:ny)

      integer :: ierr, is

c     C interface (ISO_C_BINDING)
      external :: nn_predict_sources

c     --- Pack plasma state ---
c     Te, Ti are already 2D
      Te_2d = teb
      Ti_2d = tib
c     ne = na(:,:,0,0), ni = na(:,:,0,1), ua = ua(:,:,0,1)
c     For D2-only: species 0 = electrons, species 1 = D+
      ne_2d = na(:,:,0,0)
      ni_2d = na(:,:,0,1)
      ua_2d = ua_in(:,:,0,1)

c     --- Call SOLPEx GNN via socket ---
      call nn_predict_sources(nx, ny, ns,
     &     Te_2d, Ti_2d, ne_2d, ni_2d, ua_2d,
     &     Sp_2d, Qe_2d, Qi_2d, Sm_2d,
     &     nn_socket_path, ierr)

      if (ierr .ne. 0) then
        write(*,*) '[eirene_mc_nn] ERROR: nn_predict failed, ierr=',ierr
        write(*,*) '  Keeping previous source terms unchanged.'
        return
      endif

c     --- Unpack into B2.5 source arrays ---
c     Particle source: sna0(:,:,0,is) for all ion species
c     For D2-only, main ion species is is=1 (D+)
      do is = 0, ns-1
        sna0(:,:,0,is) = 0.0_R8
        sna0(:,:,1,is) = 0.0_R8
      enddo
      sna0(:,:,0,1) = Sp_2d   ! D+ particle source

c     Electron energy source
      she0(:,:,0) = Qe_2d
      she0(:,:,1) = 0.0_R8
      she0(:,:,2) = 0.0_R8
      she0(:,:,3) = 0.0_R8

c     Ion energy source
      shi0(:,:,0) = Qi_2d
      shi0(:,:,1) = 0.0_R8
      shi0(:,:,2) = 0.0_R8
      shi0(:,:,3) = 0.0_R8

c     Momentum source: parallel component for D+
      do is = 0, ns-1
        smo0(:,:,0,is) = 0.0_R8
        smo0(:,:,1,is) = 0.0_R8
        smo0(:,:,2,is) = 0.0_R8
        smo0(:,:,3,is) = 0.0_R8
      enddo
      smo0(:,:,0,1) = Sm_2d   ! D+ parallel momentum source

      return
      end subroutine eirene_mc_nn

c-----------------------------------------------------------------------
c     eirene_nn_init: read NN configuration from b2mn.dat
c-----------------------------------------------------------------------
      subroutine eirene_nn_init()

      implicit none

      call ipgeti('b2mndr_eirene_nn', use_nn)
      call ipgetc('b2mndr_nn_socket', nn_socket_path)

      if (use_nn .ne. 0) then
        write(*,*) '================================================='
        write(*,*) ' EIRENE replaced by SOLPEx Neural Network'
        write(*,*) ' Socket: ', trim(nn_socket_path)
        write(*,*) '================================================='
      endif

      return
      end subroutine eirene_nn_init

c-----------------------------------------------------------------------
c     eirene_nn_finalize: close socket connection
c-----------------------------------------------------------------------
      subroutine eirene_nn_finalize()

      implicit none
      external :: nn_disconnect

      if (use_nn .ne. 0) then
        call nn_disconnect()
      endif

      return
      end subroutine eirene_nn_finalize

      end module b2mod_eirene_nn_mod
