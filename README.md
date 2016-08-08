### Data Description

Data is from:
> Jadhav, S. P., Rothschild, G., Roumis, D. K. & Frank, L. M. Coordinated Excitation and Inhibition of Prefrontal Ensembles during Awake Hippocampal Sharp-Wave Ripple Events. Neuron 90, 113–127 (2016).

### Raw Data Format
Data is in the Matlab format (\*.mat files). The following describes the main data structures contained in the raw data.

#### Animal Information ####
    - Name: animaldef.m
    - Description: Function that maps *animal names* to *directories*
#### Task Information
    - Name: {*animal*}task{*day*}.mat
    - Description: Defines the different task epochs for a single experimental session
    - Format:
        - 1 x {*Number of days*} Matlab-cell
            - 1 x {*Number of epochs*} Matlab-cell:
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
    - Description: Defines basic measured characteristics of each neuron such as Spike Width and Mean Rate.
    - Format:
        - 1 x {*Number of days*} Matlab-cell
            - 1 x {*Number of epochs*} Matlab-cell
                - 1 x {*Number of tetrodes*} Matlab-cell
                    - 1 x {*Number of cells*} Matlab-cell
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
    - Description: Gives basic information about each tetrode such as the depth and number of neurons recorded
    - Format:
        - 1 x {*Number of days*} Matlab-cell
            - 1 x {*Number of epochs*} Matlab-cell
                - 1 x {*Number of tetrodes*} Matlab-cell
                    - Tetrode Matlab-structure
                        - depth - depth of electrode
                        - numcells - number of neurons recorded
                        - descrip - riptet? sometimes blank
                        - area - brain area
#### Spike Information ####
    - Name: {*animal*}spikes{*day*}.mat
    - Description: Gives the spike times and other relevant information at those spike times
    - Format:
        - 1 x {*Number of days*} Matlab-cell
            - 1 x {*Number of epochs*} Matlab-cell
                - 1 x {*Number of tetrodes*} Matlab-cell
                    - Spike Matlab-structure
                        - data
                        - descript
                        - fields - fields of data array
                            - time - time of spike
                            - x - x-position of mouse at spike
                            - y - y-position of mouse at spike
                            - dir - head direction at spike
                            - not_used - ???
                            - amplitude  - amplitude of highest variance channel
                            - posindex - ???
                            - x-sm - ???
                            - y-sm - ???
                            - dir-sm - ???
                        - depth
                        - spikewidth
                        - timerange
                        - tag - brain area (sort of)
                    - cmperpixiel - centimeters per pixel?
#### EEG information (aka LFP Information) ####
    - Name: {*animal*}eeg{*day*}-{*epoch*}-{*tetrode*}.mat
    - Description: Gives the LFP for a given tetrode
    - Format:
        - 1 x {*Number of epochs*} Matlab-cell
            - 1 x {*Number of tetrodes*} Matlab-cell
                - EEG Matlab-structure
                    - descript
                    - fields - fields of data array
                    - starttime
                    - samprate
                    - data
                    - depth
#### EEG ground information (aka LFP Ground Information) ####
    - Ground wire located in the corpus collosum
    - Name: {*animal*}eeggrnd{*day*}-{*epoch*}-{*tetrode*}.mat
    - Description: Gives the LFP for a given ground tetrode
    - Format:
        - 1 x {*Number of epochs*} Matlab-cell
            - 1 x {*Number of tetrodes*} Matlab-cell
                - EEG Matlab-structure
                    - data
                    - starttime
                    - samprate
                    - depth
                    - fields
                    - descript
