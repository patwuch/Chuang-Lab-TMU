-----------------------------------------------------------------------
 Informations of denoised cluster files.                               
-----------------------------------------------------------------------
1. unique_sequence : Dereplicated unique sequence.                     
2. abundance       : Abundance of dereplicated unique sequence.        
3. ASV_tag         : ASV tag which dereplicated unique sequence belong.
                    *NA mappings are unique sequences that couldn't be 
                     denoised with good enough quality or be identified
                     as chimera.                                       


-----------------------------------------------------------------------
 Warning : Abundance of dereplicated and denoised reads doesn't match. 
-----------------------------------------------------------------------
 After denoising, reads with errors are corrected to the sequence from 
which they were inferred to orinate. Thus the denoised abundance of an 
ASV will always be greater than or equal to the dereplicated abundance 
of that sequence in the original data.

