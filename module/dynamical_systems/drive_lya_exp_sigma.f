        IMPLICIT NONE
        DOUBLE PRECISION, dimension (100) :: X
        DOUBLE PRECISION, dimension (9) :: EXPO,ZNORM, CUM, GSC
        INTEGER N,I,J,K,L,Nlya,NN,Nrun,nt
        DOUBLE PRECISION :: A,B,D,Theta,Alpha,Beta,Input,C,Sigma
        DOUBLE PRECISION :: z_MU
        DOUBLE PRECISION :: X_0,Y_0
        DOUBLE PRECISION :: Gamma
        DOUBLE PRECISION :: Rho,K1
        DOUBLE PRECISION :: K2,S,H
        COMMON/c1/N
        COMMON/c2/A,B,C,D,Theta,Alpha,Beta,Sigma,Input
        COMMON/c3/S,X_0,Y_0,z_MU
        COMMON/c4/Gamma,Rho,K1,K2

        A=3.0D0
        B=1.0D0
        D=5.0D0
        Theta=0.006D0
        Alpha = 0.1D0
        Beta = 0.02D0
        Input = 3.1D0
        C = 1.0D0
        Gamma = 3.0D0
        X_0 = -1.56D0
        Y_0 = -1.619D0
        z_MU = 0.0009D0
        S = 3.875D0
        Rho = 0.9573D0
        K1 = 1.0D0
        K2 = 1.0D0



        Open(7,File="drive_lya_sigma.txt",Status='unknown')
C
        N=5
        NN=(N*N)+N
C

        NRUN = 2000
        NLYA = 100*NRUN
        H = 0.01
c
        Sigma = 0.0D0


        DO WHILE (Sigma.LE.0.125D0)
C
C	INITIAL CONDITION FOR THE NONLINEAR SYSTEM
        X(1)=0.1D0
        X(2)=0.2D0
        X(3)=0.3D0
        X(4)=0.1D0
        X(5)=0.2D0

!
C	INITIAL CONDITION FOR LINEAR SYSTEM
        DO I = N+1,NN
        X(I)=0.0D0

        END DO
C
        DO I=1,N
        X((N+1)*I)=1.0D0
        CUM(I)=0.0D0
        END DO
C
        DO I= 1,NLYA
         CALL RK2(X,H)
        END DO
C
        DO I = 1,NLYA
          CALL RK4(X,H)

C	NORMALIZE FIRST VECTOR
        ZNORM(1)=0.0D0
        DO J=1,N
        ZNORM(1)=ZNORM(1)+X(N*J+1)**2
        END DO
        ZNORM(1)=SQRT(ZNORM(1))
        DO J=1,N
        X(N*J+1)=X(N*J+1)/ZNORM(1)
        END DO
C
C	GENERATE THE NEW ORTHONORMAL SET OF VECTORS
        DO 40 J=2,N
C
C	GENERATE J-1 GSR COEFFICIENTS
        DO 10 K=1,J-1
        GSC(K)=0.0D0
        DO 10 L=1,N
        GSC(K)=GSC(K)+X(N*L+J)*X(N*L+K)
10      CONTINUE
C
C	CONSTRUCT A NEW VECTOR
        DO 20 K=1,N
        DO 20 L=1,J-1
        X(N*K+J)=X(N*K+J)-GSC(L)*X(N*K+L)
20      CONTINUE
C
C	CALCULATE THE VECTOR'S NORM
        ZNORM(J)=0.0D0
        DO 30 K=1,N
        ZNORM(J)=ZNORM(J)+X(N*K+J)**2
30      CONTINUE
        ZNORM(J)=SQRT(ZNORM(J))
C
C	NORMALIZE THE NEW VECTOR
        DO 40 K=1,N
        X(N*K+J)=X(N*K+J)/ZNORM(J)
40      CONTINUE
C
C	UPDATE RUNNING VECTOR MAGNITUDES
        DO K=1,N
        CUM(K)=CUM(K)+DLOG(ZNORM(K))/DLOG(2.0D0)
        END DO
C
        END DO
        DO K=1,N
        EXPO(K)=CUM(K)/(H*FLOAT(NLYA))
        END DO
        WRITE(7,*)Sigma, EXPO(1), EXPO(2), EXPO(3), EXPO(4), EXPO(5)

        Sigma=Sigma+0.0001
        END DO
        STOP
        END

C	****************************************************************
        SUBROUTINE RK4(X,H)
        IMPLICIT None
        INTEGER N,I,NN
        DOUBLE PRECISION :: H
        DOUBLE PRECISION, dimension (100) :: X,TEMP,AK1,AK2,AK3
        DOUBLE PRECISION, dimension (100) :: AK4,PRIME
        COMMON/c1/N
C
        NN=N*N+N

C
        DO I=1,NN
        TEMP(I)=X(I)
        END DO
C
        CALL DERIVE(TEMP,PRIME)
        DO I=1,NN
        AK1(I)=H*PRIME(I)
        TEMP(I)=X(I)+AK1(I)/2.0D0
        END DO
C
        CALL DERIVE(TEMP,PRIME)
        DO I=1,NN
        AK2(I)=H*PRIME(I)
        TEMP(I)=X(I)+AK2(I)/2.0D0
        END DO
C
        CALL DERIVE(TEMP,PRIME)
        DO I=1,NN
        AK3(I)=H*PRIME(I)
        TEMP(I)=X(I)+AK3(I)
        END DO
C
        CALL DERIVE(TEMP,PRIME)
        DO I=1,NN
        AK4(I)=H*PRIME(I)
        X(I)=X(I)+1/6.0D0*(AK1(I)+2.0D0*(AK2(I)+AK3(I))+AK4(I))

        END DO
C
        RETURN
        END
C
C       **************************************************************
C
        SUBROUTINE DERIVE(X,PRIME)
        IMPLICIT None
        DOUBLE PRECISION, dimension (100):: X, PRIME
        INTEGER N,I
        DOUBLE PRECISION :: A,B,D,Theta,Alpha,Beta,Input,C,Sigma
        DOUBLE PRECISION :: z_MU
        DOUBLE PRECISION :: X_0,Y_0
        DOUBLE PRECISION :: Gamma
        DOUBLE PRECISION :: Rho,K1
        DOUBLE PRECISION :: K2,S,H
        COMMON/c1/N
        COMMON/c2/A,B,C,D,Theta,Alpha,Beta,Sigma,Input
        COMMON/c3/S,X_0,Y_0,z_MU
        COMMON/c4/Gamma,Rho,K1,K2

        PRIME(1) = (A * X(1)**2) - (B * X(1)**3) + X(2) - X(3) - K1*
     *  (Alpha + 3.0d0 * Beta * X(5)**2) * X(1) + Input
        PRIME(2) = C - (D * X(1) ** 2) - X(2) - Sigma * X(4)
        PRIME(3) = Theta * (S * (X(1) - X_0) - X(3))
        PRIME(4) = z_MU * (Gamma * (X(2) - Y_0) - Rho * X(4))
        PRIME(5) = X(1) - K2 * X(5)
C
        DO I= 0,N-1
        PRIME(6+I) = (2.0d0*A*X(1) - 3.0d0*B*X(1)**2 - K1*Alpha
     *  - 3.0d0*K1*Beta*X(5)**2) * X(6+I) + X(11+I) - X(16+I)
     *  - 6.0d0*K1*Beta*X(1)*X(5)*X(26+I)
        PRIME(11+I) = -2.0d0*D*X(1)*X(6+I) - X(11+I) - Sigma*X(21+I)
        PRIME(16+I) = Theta*S*X(6+I) - Theta*X(16+I)
        PRIME(21+I) = z_MU*Gamma*X(11+I) - Rho*z_MU*X(21+I)
        PRIME(26+I) = X(6+I) - K2 *X(26+I)
	    END DO
C
        RETURN
        END
C
C	****************************************************************
        SUBROUTINE RK2(X,H)
        IMPLICIT None
        INTEGER N,I
        DOUBLE PRECISION :: H
        DOUBLE PRECISION, dimension (100) :: X,TEMP,AK1,AK2,AK3
        DOUBLE PRECISION, dimension (100) :: AK4,PRIME
        COMMON/c1/N
C
        DO I=1,N
        TEMP(I)=X(I)
        END DO
C
        CALL DER(TEMP,PRIME)
        DO I=1,N
        AK1(I)=H*PRIME(I)
        TEMP(I)=X(I)+AK1(I)/2.0D0
        END DO
C
        CALL DER(TEMP,PRIME)
        DO I=1,N
        AK2(I)=H*PRIME(I)
        TEMP(I)=X(I)+AK2(I)/2.0D0
        END DO
C
        CALL DER(TEMP,PRIME)
        DO I=1,N
        AK3(I)=H*PRIME(I)
        TEMP(I)=X(I)+AK3(I)
        END DO
C
        CALL DER(TEMP,PRIME)
        DO I=1,N
        AK4(I)=H*PRIME(I)
        X(I)=X(I)+1/6.0D0*(AK1(I)+2.0D0*(AK2(I)+AK3(I))+AK4(I))
        END DO
        RETURN
        END
C
C       **************************************************************
C
        SUBROUTINE DER(X,PRIME)
        IMPLICIT None
        DOUBLE PRECISION, dimension (100):: X, PRIME
        INTEGER N
        DOUBLE PRECISION :: A,B,D,Theta,Alpha,Beta,Input,C,Sigma
        DOUBLE PRECISION :: z_MU
        DOUBLE PRECISION :: X_0,Y_0
        DOUBLE PRECISION :: Gamma
        DOUBLE PRECISION :: Rho,K1
        DOUBLE PRECISION :: K2,S,H
        COMMON/c1/N
        COMMON/c2/A,B,C,D,Theta,Alpha,Beta,Sigma,Input
        COMMON/c3/S,X_0,Y_0,z_MU
        COMMON/c4/Gamma,Rho,K1,K2

        PRIME(1) = (A * X(1)**2) - (B * X(1)**3) + X(2) - X(3) - K1*
     *  (Alpha + 3.0d0 * Beta * X(5)**2) * X(1) + Input
        PRIME(2) = C - (D * X(1) ** 2) - X(2) - Sigma * X(4)
        PRIME(3) = Theta * (S * (X(1) - X_0) - X(3))
        PRIME(4) = z_MU * (Gamma * (X(2) - Y_0) - Rho * X(4))
        PRIME(5) = X(1) - K2 * X(5)

        RETURN
        END
C

