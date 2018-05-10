# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Erney Ramirez Aportela (eramirez@cnb.csic.es)
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
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
from pyworkflow import VERSION_1_1
from pyworkflow.protocol.params import (PointerParam, StringParam, 
                                        BooleanParam, FloatParam, IntParam, LEVEL_ADVANCED)
from pyworkflow.em.protocol.protocol_3d import ProtAnalysis3D
from pyworkflow.object import Float
from pyworkflow.em import ImageHandler
from pyworkflow.utils import getExt
from pyworkflow.em.data import Volume
import numpy as np
import pyworkflow.em.metadata as md

CHIMERA_RESOLUTION_VOL = 'MG_Chimera_resolution.vol'



class XmippProtLocSharp(ProtAnalysis3D):
    """    
    Given a resolution map the protocol calculate the sharpened map.
    """
    _label = 'local Sharpening'
    _lastUpdateVersion = VERSION_1_1
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.min_res_init = Float() 
        self.max_res_init = Float()
       
    
    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('sharpSplitVolumes', BooleanParam, default=False,
                      label="Sharpening for half volumes?",
                      help='In addition to the main volume it makes sharpening for half volumes.')         
        
        form.addParam('inputVolume', PointerParam, pointerClass='Volume',
                      label="Input Map", important=True,
                      help='Select a volume for sharpening.')

        form.addParam('resolutionVolume', PointerParam, pointerClass='Volume',
                      label="Resolution Map", important=True,
                      help='Select a local resolution map.')

        form.addParam('iterations', IntParam, default=5, 
                      expertLevel=LEVEL_ADVANCED,
                      label="Iterations",
                      help='Number of iterations.')
        
        form.addParam('threads', IntParam, default=1, 
              expertLevel=LEVEL_ADVANCED,
              label="Threads",
              help='Number of threads.')
        
        form.addParam('const', FloatParam, default=1, 
                      expertLevel=LEVEL_ADVANCED,
                      label="lambda",
                      help='Regularization Param.')
  
    # --------------------------- INSERT steps functions --------------------------------------------


    def _insertAllSteps(self):
            # Convert input into xmipp Metadata format
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('sharpenStep')
        self._insertFunctionStep('createOutputStep')


    def convertInputStep(self):
        """ Read the input volume.
        """

        self.volFn = self.inputVolume.get().getFileName()
        self.resFn = self.resolutionVolume.get().getFileName()      
        extVol = getExt(self.volFn)
        extRes = getExt(self.resFn)        
        if (extVol == '.mrc') or (extVol == '.map'):
            self.volFn = self.volFn + ':mrc'
        if (extRes == '.mrc') or (extRes == '.map'):
            self.resFn = self.resFn + ':mrc'
            
        if self.sharpSplitVolumes.get() is True:
             self.vol1Fn, self.vol2Fn = self.inputMap.get().getHalfMaps()
             extVol1 = getExt(self.vol1Fn)
             extVol2 = getExt(self.vol2Fn)   
             if (extVol1 == '.mrc') or (extVol1 == '.map'):
                 self.vol1Fn = self.vol1Fn + ':mrc'    
             if (extVol2 == '.mrc') or (extVol2 == '.map'):
                 self.vol2Fn = self.vol2Fn + ':mrc'       
             
    def  sharpenStep(self):   
           
        #params = ' --vol %s' % self.volFn
        params = ' --resolution_map %s' % self.resFn
        params += ' --sampling %f' % self.inputVolume.get().getSamplingRate()
        params += ' -i %i' % self.iterations
        if (self.threads!=1):
             params += ' -n %i' % self.threads        
        if (self.const!=1):
             params += ' -l %f' % self.const
        #params += ' -o %s' % self._getExtraPath('sharpenedMap.vol')
        
        self.runJob("xmipp_volume_local_sharpening  --vol %s  -o %s"
                    %(self.volFn, self._getExtraPath('sharpenedMap.vol')), params)
        
        if self.sharpSplitVolumes:
             self.runJob("xmipp_volume_local_sharpening  --vol %s  -o %s"
                    %(self.vol1Fn, self._getExtraPath('sharpenedMap_Half1.vol')), params) 
             self.runJob("xmipp_volume_local_sharpening  --vol %s  -o %s"
                    %(self.vol2Fn, self._getExtraPath('sharpenedMap_Half2.vol')), params) 


    def createOutputStep(self):
        volume=Volume()
        volume.setFileName(self._getExtraPath('sharpenedMap.vol'))
        volume.setSamplingRate(self.inputVolume.get().getSamplingRate())

        self._defineOutputs(sharpened_map=volume)
        self._defineSourceRelation(self.inputVolume, volume)
        
        if self.sharpSplitVolumes:
            half1 = self._getExtraPath('sharpenedMap_Half1.vol')
            half2 = self._getExtraPath('sharpenedMap_Half2.vol')
            vol.setHalfMaps([half1, half2])   
            self._defineOutputs(outputVol=volume)
            self._defineSourceRelation(self.inputVolume, vol)             
            


    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        if hasattr(self, 'sharpened_map'):
            messages.append(
                'Information about the method/article in ' + MONORES_METHOD_URL)
        return messages
    
    def _summary(self):
        summary = []
        summary.append("HDsharpenedMap")
        return summary

    def _citations(self):
        return ['Ramirez-Aportela2018']

