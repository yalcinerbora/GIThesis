/**

Rectengular Prism


*/


#ifndef __RECTPRISM_H__
#define __RECTPRISM_H__

#include "IEUtility/IEVector3.h"
#include "IEUtility/IEMatrix4x4.h"
#include "IEUtility/IEMatrix3x3.h"

class IEMatrix4x4;

class RectPrism
{
	private:
		IEVector3	basis[3];		// basis vecs of the rect
									// Their length gives edge lengths
		IEVector3	cornerPoint;

	protected:
	public:
					RectPrism(const IEVector3 basis[], 
							  const IEVector3& cornerPoint);
					RectPrism(const IEVector3& aabbMin,
							  const IEVector3& aabbMax);
					~RectPrism() = default;

					
		void		Transform(const IEMatrix4x4&);
		RectPrism	Transform(const IEMatrix4x4&) const;
		void		toAABB(IEVector3& min, IEVector3& max) const;
		

	

};

inline RectPrism::RectPrism(const IEVector3 basis[],
							const IEVector3& cornerPoint)
	: cornerPoint(cornerPoint)
{
	this->basis[0] = basis[0];
	this->basis[1] = basis[1];
	this->basis[2] = basis[2];
}

inline RectPrism::RectPrism(const IEVector3& aabbMin,
							const IEVector3& aabbMax)
	: cornerPoint(aabbMin)
{
	IEVector3 diff = aabbMax - aabbMin;
	basis[0] = IEVector3::Xaxis * diff.getX();
	basis[1] = IEVector3::Yaxis * diff.getY();
	basis[1] = IEVector3::Zaxis * diff.getZ();
}

inline void RectPrism::Transform(const IEMatrix4x4& transformMatrix)
{
	// Apply Only Translate to cornerPoint
	IEVector3 translate = transformMatrix.getColumn(4);
	cornerPoint += translate;

	// Apply the rest to basis (if trasnform contains shear result is undefined)
	IEMatrix3x3 rotationScale = transformMatrix;
	basis[0] = rotationScale * basis[0];
	basis[1] = rotationScale * basis[1];
	basis[2] = rotationScale * basis[2];
}

inline RectPrism RectPrism::Transform(const IEMatrix4x4& transformMatrix) const
{
	RectPrism result(basis, cornerPoint);
	result.Transform(transformMatrix);
	return result;
}

inline void RectPrism::toAABB(IEVector3& min, IEVector3& max) const
{
	IEVector3 corners[8] = 
	{
		// Near Plane
		cornerPoint,
		cornerPoint + basis[0],
		cornerPoint + basis[1],
		cornerPoint + basis[2],

		// Far Plane
		cornerPoint + basis[0] + basis[1],
		cornerPoint + basis[0] + basis[2],
		cornerPoint + basis[1] + basis[2],
		cornerPoint + basis[0] + basis[1] + basis[2]
	};

	max.setX(std::max({corners[0].getX(),
					  corners[1].getX(),
					  corners[2].getX(),
					  corners[3].getX(),
					  corners[4].getX(),
					  corners[5].getX(),
					  corners[6].getX(),
					  corners[7].getX()}));

	max.setY(std::max({corners[0].getY(),
					  corners[1].getY(),
					  corners[2].getY(),
					  corners[3].getY(),
					  corners[4].getY(),
					  corners[5].getY(),
					  corners[6].getY(),
					  corners[7].getY()}));

	max.setZ(std::max({corners[0].getZ(),
					  corners[1].getZ(),
					  corners[2].getZ(),
					  corners[3].getZ(),
					  corners[4].getZ(),
					  corners[5].getZ(),
					  corners[6].getZ(),
					  corners[7].getZ()}));

	min.setX(std::min({corners[0].getX(),
					  corners[1].getX(),
					  corners[2].getX(),
					  corners[3].getX(),
					  corners[4].getX(),
					  corners[5].getX(),
					  corners[6].getX(),
					  corners[7].getX()}));

	min.setY(std::min({corners[0].getY(),
					  corners[1].getY(),
					  corners[2].getY(),
					  corners[3].getY(),
					  corners[4].getY(),
					  corners[5].getY(),
					  corners[6].getY(),
					  corners[7].getY()}));

	min.setZ(std::min({corners[0].getZ(),
					  corners[1].getZ(),
					  corners[2].getZ(),
					  corners[3].getZ(),
					  corners[4].getZ(),
					  corners[5].getZ(),
					  corners[6].getZ(),
					  corners[7].getZ()}));
}
#endif //__RECTPRISM_H__