# **************************************************************************
# *
# * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'jgomez@cnb.csic.es'
# *
# **************************************************************************
from pyworkflow.em.packages.xmipp3.convert import writeSetOfVolumes, volumeToRow
from pyworkflow.em.packages.xmipp3.xmipp3 import XmippMdRow
"""
This sub-package contains wrapper around reconstruct_significant Xmipp program
"""

from pyworkflow.em import *  
from convert import writeSetOfClasses2D, writeSetOfParticles

class XmippProtReconstructSignificant(ProtInitialVolume):
    """ Produces one or several initial volumes using reconstruct_significant """
    _label = 'reconstruct significant'

    #--------------------------- DEFINE param functions --------------------------------------------
    
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputClasses', PointerParam, label="Input classes", important=True, 
                      pointerClass='SetOfClasses2D, SetOfAverages',
                      help='Select the input classes2D from the project.\n'
                           'It should be a SetOfClasses2D class with class representative')
        form.addParam('symmetryGroup', TextParam, default='c1',
                      label="Symmetry group",
                      help='See [[http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Symmetry][Symmetry]]'
                      'for a description of the symmetry groups format, If no symmetry is present, give c1.')  
        form.addParam('thereisRefVolume', BooleanParam, default=False,
                      label="Is there a reference volume(s)?", 
                       help='You may use a reference volume to initialize the calculations. For instance, '
                            'this is very useful to obtain asymmetric volumes from symmetric references. The symmetric '
                            'reference is provided as starting point, choose no symmetry group (c1), and reconstruct_significant'
                            'will tend to break the symmetry finding a suitable volume. The reference volume can also be useful, '
                            'for instance, when reconstructing a fiber. Provide in this case a cylinder of a suitable size.')
        form.addParam('refVolume', PointerParam, label='Initial 3D reference volumes',
                      pointerClass='SetOfVolumes, Volume', condition="thereisRefVolume")
        form.addParam('Nvolumes', IntParam, label='Number of volumes', help="Number of volumes to reconstruct",
                      default=1,condition="not thereisRefVolume")
        form.addParam('angularSampling', FloatParam, default=5, expertLevel=LEVEL_ADVANCED,
                      label='Angular sampling',
                      help='Angular sampling in degrees for generating the projection gallery.')
        form.addParam('minTilt', FloatParam, default=0, expertLevel=LEVEL_ADVANCED,
                      label='Minimum tilt (deg)',
                      help='Use the minimum and maximum tilts to limit the angular search. This can be useful, for instance, '
                           'in the reconstruction of fibers from side views. 0 degrees is a top view, while 90 degrees is a side view.')
        form.addParam('maxTilt', FloatParam, default=90, expertLevel=LEVEL_ADVANCED,
                      label='Maximum tilt (deg)',
                      help='Use the minimum and maximum tilts to limit the angular search. This can be useful, for instance, '
                           'in the reconstruction of fibers from side views. 0 degrees is a top view, while 90 degrees is a side view.')
        form.addParam('maximumShift', FloatParam, default=-1, expertLevel=LEVEL_ADVANCED,
                      label='Maximum shift (px):', help="Set to -1 for free shift search")
        form.addParam('keepIntermediate', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label='Keep intermediate volumes',
                      help='Keep all volumes and angular assignments along iterations')

        form.addSection(label='Criteria')
        form.addParam('alpha0', FloatParam, default=80,
                      label='Starting significance',
                      help='80 means 80% of significance. Use larger numbers to relax the starting significance and have a smoother '
                           'landscape of solutions')
        form.addParam('iter', IntParam, default=100,
                      label='Number of iterations',
                      help='Number of iterations to go from the initial significance to the final one')
        form.addParam('alphaF', FloatParam, default=99.5,
                      label='Final significance',
                      help='99.5 means 99.5% of significance. Use smaller numbers to be more strict and have a sharper reconstruction.'
                           'Be aware that if you are too strict, you may end with very few projections and the reconstruction becomes very'
                           'noisy.')
        form.addParam('useImed', BooleanParam, default=True, expertLevel=LEVEL_ADVANCED,
                      label='Use IMED', help='Use IMED for the weighting. IMED is an alternative to correlation that can '
                      'discriminate better among very similar images')
        form.addParam('strictDir', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label='Strict direction', help='If the direction is strict, then only the most significant experimental images '
                      'can contribute to it. As a consequence, many experimental classes are lost and only the best contribute to the 3D '
                      'reconstruction. Be aware that only the best can be very few depending on the cases.')
        form.addParam('angDistance', IntParam, default=10, expertLevel=LEVEL_ADVANCED,
                      label='Angular neighborhood', help='Images in an angular neighborhood also determines the weight of each image. '
                      'It should be at least twice the angular sampling')
        form.addParam('dontApplyFisher', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label='Do not apply Fisher', help="Images are preselected using Fisher's confidence interval on the correlation "
                      "coefficient. Check this box if you do not want to make this preselection.")

        form.addParallelSection(threads=1, mpi=4)
    
    #--------------------------- INSERT steps functions --------------------------------------------
    
    def _insertAllSteps(self):
        """ Mainly prepare the command line for calling reconstruct_significant program"""
        
        # Convert input images if necessary
        self.imgsFn = self._getExtraPath('input_classes.xmd')
        self._insertFunctionStep('convertInputStep', self.imgsFn)

        # Prepare arguments to call program: xmipp_classify_CL2D
        self._params = {'imgsFn': self.imgsFn, 
                        'extraDir': self._getExtraPath(),
                        'symmetryGroup': self.symmetryGroup.get(),
                        'angularSampling': self.angularSampling.get(),
                        'minTilt': self.minTilt.get(),
                        'maxTilt': self.maxTilt.get(),
                        'maximumShift': self.maximumShift.get(),
                        'alpha0': 1-self.alpha0.get()/100.0,
                        'alphaF': 1-self.alphaF.get()/100.0,
                        'iter': self.iter.get(),
                        'angDistance': self.angDistance.get()
                        }
        args = '-i %(imgsFn)s --odir %(extraDir)s --sym %(symmetryGroup)s --angularSampling %(angularSampling)f '\
               '--minTilt %(minTilt)f --maxTilt %(maxTilt)f --maxShift %(maximumShift)f --iter %(iter)d --alpha0 %(alpha0)f '\
               '--alphaF %(alphaF)f --angDistance %(angDistance)f'% self._params
        
        if self.thereisRefVolume:
            args += " --initvolumes " + self._getExtraPath('input_volumes.xmd')
        else:
            args += " --numberOfVolumes %d"%self.Nvolumes.get()
        if self.useImed:
            args += " --useImed"
        if self.strictDir:
            args += " --strictDirection"
        if self.dontApplyFisher:
            args += " --dontApplyFisher"
        if self.keepIntermediate:
            args += " --keepIntermediateVolumes"
        self._insertRunJobStep("xmipp_reconstruct_significant", args)

        self._insertFunctionStep('createOutputStep')        

    #--------------------------- STEPS functions --------------------------------------------        
    def convertInputStep(self, classesFn):
        inputClasses = self.inputClasses.get()
        
        if isinstance(inputClasses, SetOfClasses2D):
            writeSetOfClasses2D(inputClasses, classesFn, writeParticles=False)
        else:
            writeSetOfParticles(inputClasses, classesFn)
            
        if self.thereisRefVolume:
            inputVolume= self.refVolume.get()
            fnVolumes = self._getExtraPath('input_volumes.xmd')
            if isinstance(inputVolume, SetOfVolumes):
                writeSetOfVolumes(inputVolume, fnVolumes)
            else:
                row = XmippMdRow()
                volumeToRow(inputVolume, row, alignType = ALIGN_NONE)
                md = xmipp.MetaData()
                row.writeToMd(md, md.addObject())
                md.write(fnVolumes)

    def getLastIteration(self,Nvolumes):
        lastIter=-1
        for n in range(self.iter.get()+1):
            NvolumesIter=len(glob(self._getExtraPath('volume_iter%03d_*.vol'%n)))
            if NvolumesIter==0:
                continue
            elif NvolumesIter==Nvolumes:
                lastIter=n
            else:
                break
        return lastIter

    def createOutputStep(self):
        if self.thereisRefVolume:
            inputVolume= self.refVolume.get()
            if isinstance(inputVolume, SetOfVolumes):
                Nvolumes=inputVolume.getSize()
            else:
                Nvolumes=1
        else:
            Nvolumes=self.Nvolumes.get()

        lastIter=self.getLastIteration(Nvolumes)
        if Nvolumes==1:
            vol = Volume()
            vol.setLocation(self._getExtraPath('volume_iter%03d_00.vol'%lastIter))
            vol.setSamplingRate(self.inputClasses.get().getSamplingRate())
        else:
            vol = self._createSetOfVolumes()
            vol.setSamplingRate(self.inputClasses.get().getSamplingRate())
            fnVolumes=glob(self._getExtraPath('volume_iter%03d_*.vol')%lastIter)
            fnVolumes.sort()
            for fnVolume in fnVolumes:
                aux=Volume()
                aux.setLocation(fnVolume)
                vol.append(aux)

        self._defineOutputs(outputVol=vol)
        self._defineSourceRelation(vol, self.inputClasses)

    #--------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []
        if self.thereisRefVolume:
            if self.refVolume.hasValue():
                refVolume = self.refVolume.get()
                x1, y1, _ = refVolume.getDim()
                x2, y2, _ = self.inputClasses.get().getDimensions()
                if x1!=x2 or y1!=y2:
                    errors.append('The input images and the reference volume have different sizes') 
            else:
                errors.append("Please, enter a reference image")
        return errors
        
    def _summary(self):
        summary = []
        summary.append("Input classes: %s" % self.inputClasses.get().getNameId())
        if self.thereisRefVolume:
            summary.append("Starting from: %s" % self.refVolume.get().getNameId())
        else:
            summary.append("Starting from: %d random volumes"%self.Nvolumes.get())
        summary.append("Significance from %f%% to %f%% in %d iterations"%(self.alpha0.get(),self.alphaF.get(),self.iter.get()))
        if self.useImed:
            summary.append("IMED used")
        if self.strictDir:
            summary.append("Strict directions")    
        return summary
    
    def _citations(self):
        return ['Sorzano2015']
    
    def _methods(self):
        retval="We used reconstruct significant to produce an initial volume from the set of classes %s."%\
           self.inputClasses.get().getNameId()
        if self.thereisRefVolume:
            retval+=" We used "+self.refVolume.get().getNameId()+" as a starting point of the reconstruction iterations."
        else:
            retval+=" We started the iterations with %d random volumes."%self.Nvolumes.get()
        retval+=" %d iterations were run going from a starting significance of %f%% to a final one of %f%%."%\
           (self.iter.get(),self.alpha0.get(),self.alphaF.get())
        if self.useImed:
            retval+=" IMED weighting was used."
        if self.strictDir:
            retval+=" The strict direction criterion was employed."    
        return [retval]
