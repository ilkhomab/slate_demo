subroutine solveRK_num_1c_dev( rho,prob,zamo )
   use precision_type
   use data_base,only:num_t,vleft,vmid,vright,nstep,step,i0,unitarity2,cu,nglag,ngleg,u,glag,wig_d,lmax_t,mmax_t,fdir,idir,jdir,k_t,inl_t,zam,zout,zdadz,dyt,dym,dymm
   use acc_routines,only:direct_num_t
   implicit none
   integer::ist_f,i,j,ist_i,k
   integer:: fin,iin,finl,iinl
   real(kind=id),intent(in) :: rho
   real(kind=id),intent(inout),dimension(1:num_t) :: prob
!   complex(kind=id),dimension(1:num_t) :: zam,zout,zdadz,dyt,dym,dymm
!   complex(kind=id) :: tmp 
   complex(kind=id),intent(out),dimension(1:num_t) :: zamo
   real(kind=id) :: z,znew,h,zmid,bigr,cosang,hh,h6
   real(kind=id) :: kf,ki
   integer :: t1, t2, dt, count_rate, count_max
   real(kind=id) ::  secs

!    open(220,file='solution',status='unknown')
!    write(220,*)
!    write(220,*)

allocate(zam(1:num_t))
allocate(zout(1:num_t))
allocate(zdadz(1:num_t))
allocate(dyt(1:num_t))
allocate(dym(1:num_t))
allocate(dymm(1:num_t))

print*,'I am here'
   call system_clock(count_max=count_max, count_rate=count_rate)
      call system_clock(t1)

do ist_f=1,num_t
   zam(ist_f)=cmplx(0,0,id)
   zdadz(ist_f)=cmplx(0,0,id)
   zout(ist_f)=cmplx(0,0,id)
enddo
zam(i0)=cmplx(1,0,id)
z=step(-nstep)
bigr=sqrt(rho*rho+z*z)
cosang=z/bigr
do j=1,jdir
  vright(fdir(j),idir(j))=direct_num_t(fdir(j),idir(j),cosang,bigr)*cmplx(cos(z*(k_t(inl_t(idir(j)))-k_t(inl_t(fdir(j))))),sin(z*(k_t(inl_t(idir(j)))-k_t(inl_t(fdir(j))))),id)
  vright(idir(j),fdir(j))=conjg(vright(fdir(j),idir(j)))
enddo

do j=1,num_t
  vright(j,j)=direct_num_t(j,j,cosang,bigr)*cmplx(cos(z*(k_t(inl_t(j))-k_t(inl_t(j)))),sin(z*(k_t(inl_t(j))-k_t(inl_t(j)))),id)
enddo

do i=-nstep,nstep-1

   z=step(i)

   znew=step(i+1)
   zmid=(z+znew)/real(2,id)
   h=znew-z
   hh=h/2._id
   h6=h/6._id

   do k=1,num_t 
     vleft(k,1:num_t)=vright(k,1:num_t)
     zdadz(k)=-cu*sum(vleft(k,1:num_t)*zam(1:num_t))
   enddo

   bigr=sqrt(rho*rho+zmid*zmid)
   cosang=zmid/bigr

   do j=1,jdir
     vmid(fdir(j),idir(j))=direct_num_t(fdir(j),idir(j),cosang,bigr)*cmplx(cos(zmid*(k_t(inl_t(idir(j)))-k_t(inl_t(fdir(j))))),sin(zmid*(k_t(inl_t(idir(j)))-k_t(inl_t(fdir(j))))),id)
     vmid(idir(j),fdir(j))=conjg(vmid(fdir(j),idir(j)))
   enddo
   do j=1,num_t
     vmid(j,j)=direct_num_t(j,j,cosang,bigr)*cmplx(cos(z*(k_t(inl_t(j))-k_t(inl_t(j)))),sin(z*(k_t(inl_t(j))-k_t(inl_t(j)))),id)
   enddo

   bigr=sqrt(rho*rho+znew*znew)
   cosang=znew/bigr

   do j=1,jdir
     vright(fdir(j),idir(j))=direct_num_t(fdir(j),idir(j),cosang,bigr)*cmplx(cos(znew*(k_t(inl_t(idir(j)))-k_t(inl_t(fdir(j))))),sin(znew*(k_t(inl_t(idir(j)))-k_t(inl_t(fdir(j))))),id)
     vright(idir(j),fdir(j))=conjg(vright(fdir(j),idir(j)))
   enddo
   do j=1,num_t
     vright(j,j)=direct_num_t(j,j,cosang,bigr)*cmplx(cos(z*(k_t(inl_t(j))-k_t(inl_t(j)))),sin(z*(k_t(inl_t(j))-k_t(inl_t(j)))),id)
   enddo

   do j=1,num_t 
     dyt(j)=-cu*sum(vmid(j,1:num_t)*(zam(1:num_t)+hh*zdadz(1:num_t)))
   enddo
   do j=1,num_t 
     dym(j)=-cu*sum(vmid(j,1:num_t)*(zam(1:num_t)+hh*dyt(1:num_t)))
   enddo
   do j=1,num_t 
     dymm(j)=dyt(j)+dym(j)
     dyt(j)=-cu*sum(vright(j,1:num_t)*(zam(1:num_t)+h*dym(1:num_t)))
     zout(j)=zam(j)+h6*(zdadz(j)+dyt(j)+2._id*dymm(j))
   enddo
   do j=1,num_t 
     zam(j)=zout(j)
   enddo

enddo

unitarity2=sum(abs(zam(1:num_t))**2)

    zam(i0)=zam(i0)-1._id

    do ist_f=1,num_t
       prob(ist_f)=zam(ist_f)*conjg(zam(ist_f))*rho
    enddo

zamo(:)=zam(:)

    write(1298,*)rho,unitarity2

      call system_clock(t2)
      dt = t2-t1
      secs = real(dt)/real(count_rate)

      open(1299,file='time',status='unknown')
      write(1299,"('number of channels is ',i5,' wall clock time is ',f12.2,' seconds')") num_t,secs
      open(1300,file='timing',status='unknown')
      write(1300,"(i5,f12.2)") num_t,secs

 30   format(i20,10000g14.6)

deallocate(zam)
deallocate(zout)
deallocate(zdadz)
deallocate(dyt)
deallocate(dym)
deallocate(dymm)
end subroutine solveRK_num_1c_dev
