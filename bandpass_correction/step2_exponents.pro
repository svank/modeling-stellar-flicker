;---------------------------------------------------------------------------
; The output from step1_fluxes.pro needs to be collected into separate
; files for each discrete value of log g.  This has been done with the
; files in this directory called "stars_logg***.out"
;
; Once those are in place, this IDL procedure chooses one of them to read
; in (depending on which FILE01 name is not commented out) and it computes
; the derivative in T_eff space, the exponent "m", & the C_BP multiplier.
;
; Written by S. R. Cranmer, February 2020
;---------------------------------------------------------------------------

; read in data

;   FILE01 = 'stars_logg1.5.out'
;   FILE01 = 'stars_logg2.0.out'
;   FILE01 = 'stars_logg2.5.out'
;   FILE01 = 'stars_logg3.0.out'
;   FILE01 = 'stars_logg3.5.out'
    FILE01 = 'stars_logg4.0.out'
;   FILE01 = 'stars_logg4.5.out'
;   FILE01 = 'stars_logg5.0.out'

    get_lun,unit & openr,unit,FILE01 & readf,unit,npts
    Teff = dblarr(npts)
    logg = dblarr(npts)
    Lbol   = dblarr(npts)
    Lkep   = dblarr(npts)
    for i=0,npts-1 do begin
      readf,unit,x1,x2,x3,x4
      Teff(i)   = x1
      logg(i)   = x2
      Lbol(i)   = x3
      Lkep(i)   = x4
    endfor
    close,unit & free_lun,unit

; compute exponent and C_{BP} bandpass-correction multiplier, and output

    exponent   = deriv(alog(Teff),alog(Lkep))
    sigma_mult = exponent*(exponent-1.)/12.

    for i=0,npts-1 do begin
      if ((Teff(i) ge 3500.) and (Teff(i) le 9000.)) then $
        print,Teff(i),logg(i),sigma_mult(i), $
           format='(f7.1,f7.3,e16.8)'
    endfor

    end
