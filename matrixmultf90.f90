program main
    use omp_lib
    implicit none 
    integer ::  matARow
    integer :: matACol
    integer :: matBRow
    integer :: matBCol
    real :: startTime
    real :: endTime
    integer :: numberOfThreads

    real, allocatable, dimension(:,:) :: matrixA
    real, allocatable, dimension(:,:) :: matrixB
 
    real, allocatable, dimension(:,:) :: matrixC    

    call random_seed()
    ! setbuf(stdout, NULL);
    print *, "Hello!"

    print *, "Please enter the number of rows of matrix A"
    read *, matARow

    print *, 'Please enter the number of columns of matrix A'
    read *, matACol

    print *, 'Please enter the number of rows of matrix B'
    read *, matBRow

    print *, 'Please enter the number of columns of matrix B'
    read *, matBCol
 
    allocate(matrixA(matACol, matARow))
    allocate(matrixB(matBCol, matBRow))

    if (matACol /= matBRow) then
        print *, 'Dimensions incompatible for matrix multiplication'
        stop
    end if
    
    startTime = omp_get_wtime()
    call normalmatrixmultiply(matARow, matACol, matBRow, matBCol, matrixA, matrixB)
    endTime = omp_get_wtime()
    print *, 'The time for normal matrix multiplication is: ', endTime - startTime

    startTime = omp_get_wtime()
    call ompmatrixmultiply(matARow, matACol, matBRow, matBCol, matrixA, matrixB, numberOfThreads)
    endTime = omp_get_wtime()
    print *, 'The time for omp matrix multiplication with', numberOfThreads, ' is: ', endTime - startTime

    deallocate (matrixA)
    deallocate (matrixB)

CONTAINS

subroutine normalmatrixmultiply(matARow, matACol, matBRow, matBCol, matrixA, matrixB)
    ! This is a comment line; ignored by the compiler
    implicit none
    integer, intent(in) :: matARow, matACol, matBRow, matBCol
    real, allocatable, dimension(:,:), intent(inout) :: matrixA, matrixB
    integer :: i
    integer :: j
    integer :: k
 
    real, allocatable, dimension(:,:) :: matrixC
   
    ! setbuf(stdout, NULL);
   
    do i = 1, matARow
        do j = 1, matACol
            !print *, 'Please enter the element at row', i, 'column', j, 'of matrix A\n'
            !read *, matrixA(j,i)
            call random_number(matrixA(j,i))
        end do
    end do

    do i = 1, matBRow
        do j = 1, matBCol
            !print *, 'Please enter the element at row', i, 'column', j, 'of matrix B\n'
            !read *, matrixB(j,i)
            call random_number(matrixB(j,i))
        end do
    end do

    allocate(matrixC(matBCol, matARow))
    do i = 1, matARow
        do j = 1, matBCol
            do k = 1, matACol
                matrixC(j,i) = matrixC(j,i) + matrixA(k,i) * matrixB(j,k)
            enddo
       enddo
    enddo

    !do i = 1, matARow
    !    do j = 1, matBCol
    !            if (j == matARow) then
    !                print *, " ", matrixC(j,i), '\n' 
    !            else
    !                print *, " ", matrixC(j,i)
    !            end if
    !    enddo
    !enddo

    deallocate (matrixC)

end subroutine normalmatrixmultiply


subroutine ompmatrixmultiply(matARow, matACol, matBRow, matBCol, matrixA, matrixB, numberOfThreads)
    ! This is a comment line; ignored by the compiler
    implicit none
    integer, intent(in) :: matARow, matACol, matBRow, matBCol
    integer, intent(inout) :: numberofThreads
    real, allocatable, dimension(:,:), intent(inout) :: matrixA, matrixB
    integer :: i
    integer :: j
    integer :: k
    real :: tmp
 
    real, allocatable, dimension(:,:) :: matrixC
   
    ! setbuf(stdout, NULL);
    !$OMP PARALLEL DEFAULT(none) SHARED(matrixA,matrixB,matrixC,matARow,matACol,matBRow,matBCol,numberOfThreads) PRIVATE(i,j,k,tmp)
    numberOfThreads = omp_get_num_threads();
    !$OMP DO SCHEDULE(GUIDED) COLLAPSE(2)
        do i = 1, matARow
            do j = 1, matACol
                !print *, 'Please enter the element at row', i, 'column', j, 'of matrix A\n'
                !read *, matrixA(j,i)
                call random_number(matrixA(j,i))
            end do
        end do
    !$OMP END DO

    !$OMP DO SCHEDULE(GUIDED) COLLAPSE(2)
        do i = 1, matBRow
            do j = 1, matBCol
                !print *, 'Please enter the element at row', i, 'column', j, 'of matrix B\n'
                !read *, matrixB(j,i)
                call random_number(matrixB(j,i))
            end do
        end do
    !$OMP END DO
    !$OMP END PARALLEL

    allocate(matrixC(matBCol, matARow))

    !$OMP PARALLEL DEFAULT(none) SHARED(matrixA,matrixB,matrixC,matARow,matBRow,matACol,matBCol,numberOfThreads) PRIVATE(i,j,k,tmp)
        !$OMP DO SCHEDULE(GUIDED) COLLAPSE(2)
        do i = 1, matARow
            do j = 1, matBCol
                do k = 1, matACol
                    tmp = tmp + matrixA(k,i) * matrixB(j,k)
                enddo
                matrixC(j,i) = tmp
           enddo
        enddo
        !$OMP END DO
        
        !$OMP BARRIER

        !!$OMP DO SCHEDULE(GUIDED) COLLAPSE(2)
        !do i = 1, matARow
        !     do j = 1, matBCol
        !            if (j == matARow) then
        !                print *, " ", matrixC(j,i), '\n' 
        !            else
        !                print *, " ", matrixC(j,i)
        !            end if
        !    enddo
        !enddo
        !!$OMP END DO
    !$OMP END PARALLEL

    deallocate (matrixC)

end subroutine ompmatrixmultiply

end program main
