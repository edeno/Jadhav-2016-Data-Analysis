### Data Description

Data is from:
> Jadhav, S. P., Rothschild, G., Roumis, D. K. & Frank, L. M. Coordinated Excitation and Inhibition of Prefrontal Ensembles during Awake Hippocampal Sharp-Wave Ripple Events. Neuron 90, 113–127 (2016).

### Raw Data Format
Data is in the Matlab format (\*.mat files)

#### Animal Information ####
    - Name: animaldef.m
    - Function that maps *animal names* to *directories*
#### Task Information
    - Name: {*animal*}task{*day*}.mat
    - Describes the task epochs
    - 1 x {*\# of days*} Matlab-cell
        - 1 x {*\# of epochs*} Matlab-cell:
            - task epoch Matlab-structure
                - Type - type of epoch (sleep, run, rest)
                - Linear coord: linearized positional coordinates? (only there if relevant to task epoch)
                - Environment: type of running track
                    -  lin: linear track
                    -  wtr1: w-track
                    -  postsleep
                    -  presleep
#### Cell Information (aka Neuron Information) ####
    - Name: {*animal*}cellinfo.mat
    - 1 x {*\# of days*} Matlab-cell
        - 1 x {*\# of epochs*} Matlab-cell
            - 1 x {*\# of tetrodes*} Matlab-cell
                - 1 x {*\# of cells*} Matlab-cell
                    - Cell Matlab-structure
                        - spikewidth
                        - meanrate
                        - numspikes
                        - csi
                        - propbursts
                        - tag
                        - descip
                        - area - brain area
                        - tag2
                        - ...
#### Tetrode Information ####
    - Name: {*animal*}tetinfo.mat
    - 1 x {*\# of days*} Matlab-cell
        - 1 x {*\# of epochs*} Matlab-cell
            - 1 x {*\# of tetrodes*} Matlab-cell
                - Tetrode Matlab-structure
                    - depth - depth of electrode
                    - numcells - number of cells recorded
                    - descrip - riptet? sometimes blank
                    - area - brain area
#### Spike Information ####
    - Name: {*animal*}spikes{*day*}
    - 1 x {*\# of days*} Matlab-cell
        - 1 x {*\# of epochs*} Matlab-cell
            - 1 x {*\# of tetrodes*} Matlab-cell
                - Spike Matlab-structure
                    - data
                    - descript - spike data
                    - fields - of the data field
                        - time - time of spike?
                        - x - x-position of mouse?
                        - y - y-position of mouse?
                        - dir - head direction?
                        - not_used?
                        - amplitude  - of highest variance channel
                        - posindex - ?
                        - x-sm - ?
                        - y-sm - ?
                        - dir-sm - ?
                    - depth
                    - spikewidth
                    - timerange
                    - tag - brain area?
                - cmperpixiel - centimeters per pixel?
#### EEG information (aka LFP Information) ####
    - Name: {*animal*}eeg{*day*}-{*epoch*}-{*tetrode*}
        - 1 x {*\# of epochs*} Matlab-cell
            - 1 x {*\# of tetrodes*} Matlab-cell
                - EEG Matlab-structure
                    - descript
                    - fields - fields of data struct
                    - starttime
                    - samprate
                    - data
                    - depth
#### EEG ground information (aka LFP Ground Information) ####
    - Ground wire located in the corpus collosum
    - Name: {*animal*}eeggrnd{*day*}-{*epoch*}-{*tetrode*}.mat
        - 1 x {*\# of epochs*} Matlab-cell
            - 1 x {*\# of tetrodes*} Matlab-cell
                - EEG Matlab-structure
                    - data
                    - starttime
                    - samprate
                    - depth
                    - fields
                    - descript
