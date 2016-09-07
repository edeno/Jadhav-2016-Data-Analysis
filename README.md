### Data Description

Data is from:
> Jadhav, S. P., Rothschild, G., Roumis, D. K. & Frank, L. M. Coordinated Excitation and Inhibition of Prefrontal Ensembles during Awake Hippocampal Sharp-Wave Ripple Events. Neuron 90, 113–127 (2016).

### Raw Data Format
Data is in the Matlab format (.mat files). The following describes the main data structures contained in the raw data.

#### Animal Information ####
- **Name**: animaldef.m
- **Description**: Function that maps *animal names* to *directories*

#### Task Information
- **Name**: {*animal*}task{*day*}.mat
- **Description**: Defines the different task epochs for a single experimental session
- **Format**:
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
- **Name**: {*animal*}cellinfo.mat
- **Description**: Defines basic measured characteristics of each neuron such as Spike Width and Mean Rate.
- **Format**:
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
- **Name**: {*animal*}tetinfo.mat
- **Description**: Gives basic information about each tetrode such as the depth and number of neurons recorded, lists the valid electrodes
- **Format**:
    - 1 x {*Number of days*} Matlab-cell
        - 1 x {*Number of epochs*} Matlab-cell
            - 1 x {*Number of tetrodes*} Matlab-cell
                - Tetrode Matlab-structure
                    - depth: depth of electrode
                    - numcells: number of neurons recorded
                    - descrip: riptet? sometimes blank
                    - area: brain area

#### Spike Information ####
- **Name**: {*animal*}spikes{*day*}.mat
- **Description**: Gives the spike times and other relevant information at spike times.
- **Format**:
    - 1 x {*Number of days*} Matlab-cell
        - 1 x {*Number of epochs*} Matlab-cell
            - 1 x {*Number of tetrodes*} Matlab-cell
                - Spike Matlab-structure
                    - data: Matlab-array of data described by fields
                    - descript
                    - fields: fields of data array
                        - time: time of spike event in seconds
                        - x: x-position of animal at spike
                        - y: y-position of animal at spike
                        - dir: head direction at spike
                        - not_used: ???
                        - amplitude: amplitude of highest variance channel
                        - posindex: ???
                        - x-sm: ???
                        - y-sm: ???
                        - dir-sm: ???
                    - depth
                    - spikewidth
                    - timerange - in 100 µsec units
                    - tag - brain area (sort of)
                - cmperpixiel - centimeters per pixel?

#### EEG information (aka LFP Information) ####
- **Name**: {*animal*}eeg{*day*}-{*epoch*}-{*tetrode*}.mat
- **Description**: Gives the LFP for a given tetrode.
- **Format**:
    - 1 x {*Number of epochs*} Matlab-cell
        - 1 x {*Number of tetrodes*} Matlab-cell
            - EEG Matlab-structure
                - descript - timestamps in hrs:min:sec
                - fields - fields of data array
                - starttime - in seconds
                - samprate - sampling rate
                - data - data array
                - depth - depth of electrode

#### EEG ground information (aka LFP Ground Information) ####
- **Name**: {*animal*}eeggrnd{*day*}-{*epoch*}-{*tetrode*}.mat
- **Description**: Gives the LFP for a given ground tetrode (Ground wire located in the corpus collosum)
- **Format**:
    - 1 x {*Number of epochs*} Matlab-cell
        - 1 x {*Number of tetrodes*} Matlab-cell
            - EEG Matlab-structure
                - descript - timestamps in hrs:min:sec
                - fields - fields of data array
                - starttime - in seconds
                - samprate - sampling rate
                - data - data array
                - depth - depth of electrode

#### Position Information ####
- **Name**: {*animal*}pos{*day*}.mat
- **Description**: Position of animal on track. Derived from the rawpos data structures. Timestamps are in seconds.
- **Format**:
    - 1 x {*Number of days*}
        - 1 x {*Number of epochs*}
            - position structure
                - arg: arguments used to derive the position structure from the rawpos information
                - descript: description of the data in the position structure
                - fields: labels for the data array
                - data: array with field labels
                    - time: time in session in seconds
                    - x: x-position of animal
                    - y: y-position of animal
                    - dir: head-direction of animal
                    - vel: velocity of animal (cm / s)
                - cmperpixel: - frames per second of ccd camera

#### Linearized Position Information ####
- **Name**: {*animal*}linpos{*day*}.mat
- **Description**: Gives the animal's distance from a well (either at the starting position well or at the end of one of the arms). Derived from the pos data structure.
- **Format**:

#### Digital Input Output Information ####
- **Name**: {*animal*}DIO{*day*}.mat
- **Description**: The *DIO* cell gives arrival/departure times at the end of each arm of the maze (as indicated by the IR motion sensors at the end of the wells) and the start/stop times for the output trigger to the reward pump. Timestamps are in 100 µsec units.
- **Format**:
    - 1 x {*Number of days*}
        - 1 x {*Number of epochs*}
            - 1 x {*Number of DIO-board pins*}
                - non-empty cells correspond to active pins
                - there are two active pins for each reward well corresponding to either the IR motion sensor or the reward pump
                - DIO structure
                    - pulsetimes: start/stop time of activation
                    - timesincelast: time since last activation
                    - pulselength: duration of activation
                    - pulseind: index of pulse
