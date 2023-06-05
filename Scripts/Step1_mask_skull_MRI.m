% Skull removal from MRI scans
MASK=1

logNr=['M0573';'M0572';'M0574';'M0576','M0579';'M0588';'M0593';'M0595';'M0599';'M0590';'M0594';'M0598'] ;

for i=1:12
    addpath('...\PythonScripts\MRI_processing\')
    addpath('...\MATLAB_packages\NIfTI_20140122\')
    BASE= ['...\MRI_data\Data\' logNr(i,:)];
    cd(BASE)

    % Segment brain from the TRUE-FISP scan
    if MASK
    in=dir([BASE '\DICOM2NIFTI\' '*_T2_TRUE_FISP_3D_.nii']);
    fileX =   in.name ;
    anatomy = load_untouch_nii([BASE  '\DICOM2NIFTI\' fileX]);
    anatomy_thresh = double(anatomy.img);
    minval = min(min(min(anatomy_thresh)));
    maxval = max(max(max(anatomy_thresh)));
    anatomy_thresh = ((anatomy_thresh(:,:,:)-minval))./(maxval-minval);
    anatomy_thresh = int16(round(anatomy_thresh.*255));
    anatomy2save = zeros(256,128,256);
    anatomy2save(anatomy_thresh>0)=anatomy_thresh(anatomy_thresh>0);

    I = anatomy2save;
    bw = imbinarize(I,44); %53!
    se = strel('disk',2); % 2, 3,4!,6
    bwo = uint16(imopen(bw,se));
    bwoe = imerode(bwo,se);
    %medf = medfilt3(bwo,'zeros');
    %imshow(bwo,[0 1])
    cc = bwconncomp(bwoe,6);
    numPixels = cellfun(@numel,cc.PixelIdxList);
    [biggest,idx] = max(numPixels);
    bwoe(cc.PixelIdxList{idx}) = 2;
    biggest_blob = zeros(256,128,256);
    biggest_blob(bwoe==2)=1;

    se = strel('disk',2); % 4!
    biggest_blob = uint16(imdilate(biggest_blob,se));
    biggest_blob = uint16(imfill(biggest_blob,'holes'));
    save_nii(make_nii(uint16(biggest_blob)),[BASE '\Mask.nii']);

    end

end
