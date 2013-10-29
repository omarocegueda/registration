TReal InvertField(DisplacementFieldPointer field,
                    DisplacementFieldPointer inverseField, TReal weight = 1.0,
                    TReal toler = 0.1, int maxiter = 20, bool /* print */ = false)
  {

    TReal        mytoler = toler;
    unsigned int mymaxiter = maxiter;

    typename ParserType::OptionType::Pointer thicknessOption
      = this->m_Parser->GetOption( "go-faster" );
    if( thicknessOption->GetFunction( 0 )->GetName() == "true" ||  thicknessOption->GetFunction( 0 )->GetName() == "1" )
      {
      mytoler = 0.5; maxiter = 12;
      }

    VectorType zero; zero.Fill(0);
    //  if (this->GetElapsedIterations() < 2 ) maxiter=10;

    ImagePointer TRealImage = AllocImage<ImageType>(field);

    typedef typename DisplacementFieldType::PixelType           DispVectorType;
    typedef typename DisplacementFieldType::IndexType           DispIndexType;
    typedef typename DispVectorType::ValueType                  ScalarType;
    typedef ImageRegionIteratorWithIndex<DisplacementFieldType> Iterator;

    DisplacementFieldPointer lagrangianInitCond =
      AllocImage<DisplacementFieldType>(field);
    DisplacementFieldPointer eulerianInitCond =
      AllocImage<DisplacementFieldType>(field);

    typedef typename DisplacementFieldType::SizeType SizeType;
    SizeType size = field->GetLargestPossibleRegion().GetSize();

    typename ImageType::SpacingType spacing = field->GetSpacing();

    unsigned long npix = 1;
    for( int j = 0; j < ImageDimension; j++ ) // only use in-plane spacing
      {
      npix *= field->GetLargestPossibleRegion().GetSize()[j];
      }


    TReal    max = 0;
    Iterator iter( field, field->GetLargestPossibleRegion() );
    for(  iter.GoToBegin(); !iter.IsAtEnd(); ++iter )
      {
      DispIndexType  index = iter.GetIndex();
      DispVectorType vec1 = iter.Get();
      DispVectorType newvec = vec1 * weight;
      lagrangianInitCond->SetPixel(index, newvec);
      TReal mag = 0;
      for( unsigned int jj = 0; jj < ImageDimension; jj++ )
        {
        mag += newvec[jj] * newvec[jj];
        }
      mag = sqrt(mag);
      if( mag > max )
        {
        max = mag;
        }
      }

    eulerianInitCond->FillBuffer(zero);

    TReal scale = (1.) / max;
    if( scale > 1. )
      {
      scale = 1.0;
      }
//    TReal initscale=scale;
    Iterator vfIter( inverseField, inverseField->GetLargestPossibleRegion() );

//  int num=10;
//  for (int its=0; its<num; its++)
    TReal        difmag = 10.0;
    unsigned int ct = 0;

    TReal        meandif = 1.e8;
//    int badct=0;
//  while (difmag > subpix && meandif > subpix*0.1 && badct < 2 )//&& ct < 20 && denergy > 0)
//    TReal length=0.0;
    TReal stepl = 2.;

    TReal epsilon = (TReal)size[0] / 256;
    if( epsilon > 1 )
      {
      epsilon = 1;
      }

    while( difmag > mytoler && ct<mymaxiter && meandif> 0.001 )
      {
      meandif = 0.0;

      // this field says what position the eulerian field should contain in the E domain
      this->ComposeDiffs(inverseField, lagrangianInitCond,    eulerianInitCond, 1);
      difmag = 0.0;
      for(  vfIter.GoToBegin(); !vfIter.IsAtEnd(); ++vfIter )
        {
        DispIndexType  index = vfIter.GetIndex();
        DispVectorType update = eulerianInitCond->GetPixel(index);
        TReal      mag = 0;
        for( int j = 0; j < ImageDimension; j++ )
          {
          update[j] *= (-1.0);
          mag += (update[j] / spacing[j]) * (update[j] / spacing[j]);
          }
        mag = sqrt(mag);
        meandif += mag;
        if( mag > difmag )
          {
          difmag = mag;
          }
        //      if (mag < 1.e-2) update.Fill(0);

        eulerianInitCond->SetPixel(index, update);
        TRealImage->SetPixel(index, mag);
        }
      meandif /= (TReal)npix;
      if( ct == 0 )
        {
        epsilon = 0.75;
        }
      else
        {
        epsilon = 0.5;
        }
      stepl = difmag * epsilon;
      for(  vfIter.GoToBegin(); !vfIter.IsAtEnd(); ++vfIter )
        {
        TReal      val = TRealImage->GetPixel(vfIter.GetIndex() );
        DispVectorType update = eulerianInitCond->GetPixel(vfIter.GetIndex() );
        if( val > stepl )
          {
          update = update * (stepl / val);
          }
        DispVectorType upd = vfIter.Get() + update * (epsilon);
        vfIter.Set(upd);
        }
      ct++;

      }

    // ::ants::antscout <<" difmag " << difmag << ": its " << ct <<  std::endl;

    return difmag;

  }